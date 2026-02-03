import os
import csv
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from src.helper import load_pdf, load_pdf_safe, download_hugging_face_embeddings, text_split
from langchain.schema import Document
import time
import math
import sys

# -----------------------------
# 1️⃣ Load environment variables
# -----------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "keyssss.env"))

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
INDEX_NAME = "medical-chatbot"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables!")

print("Pinecone API Key loaded successfully.")

# -----------------------------
# 2️⃣ Load ALL PDFs from both directories
# -----------------------------
print("Loading ALL PDF documents...")

# Load from data/ directory
data_docs = []
try:
    data_docs = load_pdf_safe("data/")
    print(f"✅ Loaded {len(data_docs)} documents from data/")
except Exception as e:
    print(f"❌ Error loading data/: {e}")

# Load from data2/ directory  
data2_docs = []
try:
    data2_docs = load_pdf_safe("data2/")
    print(f"✅ Loaded {len(data2_docs)} documents from data2/")
except Exception as e:
    print(f"❌ Error loading data2/: {e}")

# Combine all PDF documents
all_pdf_docs = data_docs + data2_docs
print(f"📚 Total PDF documents: {len(all_pdf_docs)}")

# Print what we found
print("\n📋 PDF Files Found:")
for doc in all_pdf_docs[:20]:  # Show first 20
    source = doc.metadata.get('source', 'unknown')
    filename = os.path.basename(source)
    print(f"  - {filename}")

if len(all_pdf_docs) > 20:
    print(f"  ... and {len(all_pdf_docs) - 20} more")

# -----------------------------
# 3️⃣ Load hospital directory CSV
# -----------------------------
hospital_csv = os.path.join('data', 'hospital_directory.csv')
hospital_docs = []
if os.path.exists(hospital_csv):
    print(f"\n🏥 Loading hospital directory from: {hospital_csv}")
    with open(hospital_csv, encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Construct a short text representation for each hospital
            name = row.get('Hospital_Name') or row.get('Hospital_Name'.upper()) or ''
            state = row.get('State') or ''
            district = row.get('District') or ''
            specialties = row.get('Specialties') or ''
            phone = row.get('Telephone') or row.get('Mobile_Number') or ''
            address = row.get('Address_Original_First_Line') or ''

            text = f"Hospital: {name}\nState: {state}\nDistrict: {district}\nAddress: {address}\nPhone: {phone}\nSpecialties: {specialties}"
            hospital_docs.append(Document(
                page_content=text, 
                metadata={
                    "source": hospital_csv, 
                    "hospital_row": i, 
                    "hospital_name": name,
                    "type": "hospital_directory"
                }
            ))

    print(f"✅ Converted {len(hospital_docs)} hospital rows into documents")
else:
    print(f"❌ hospital_directory.csv not found at: {hospital_csv}")

# -----------------------------
# 4️⃣ Combine ALL documents and split into chunks
# -----------------------------
all_documents = all_pdf_docs + hospital_docs

if not all_documents:
    raise SystemExit("❌ No documents found to index!")

print(f"\n🔪 Splitting {len(all_documents)} documents into chunks...")
text_chunks = text_split(all_documents)
print(f"✅ Created {len(text_chunks)} text chunks")

# -----------------------------
# 5️⃣ Initialize Pinecone
# -----------------------------
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    # Get embedding dimension
    test_embedding = embeddings.embed_query("test")
    dimension = len(test_embedding) if hasattr(test_embedding, '__len__') else 384
    
    pc.create_index(
        name=INDEX_NAME,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Created new Pinecone index: {INDEX_NAME} (dimension: {dimension})")
else:
    print(f"✅ Using existing Pinecone index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

# -----------------------------
# 6️⃣ Batch embedding and upsert with progress
# -----------------------------
# Optional limit for testing
INDEX_LIMIT = None
if len(sys.argv) > 1:
    try:
        INDEX_LIMIT = int(sys.argv[1])
        print(f"🧪 DEBUG: INDEX_LIMIT set to {INDEX_LIMIT}")
    except Exception:
        INDEX_LIMIT = None

total_chunks = len(text_chunks)
if INDEX_LIMIT:
    total_chunks = min(total_chunks, INDEX_LIMIT)

print(f"\n🚀 Starting embedding process for {total_chunks} chunks...")

BATCH_SIZE = 50
RETRY_ATTEMPTS = 3
start_time = time.time()
upserted = 0

for start in range(0, total_chunks, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total_chunks)
    batch_docs = text_chunks[start:end]
    texts = [d.page_content for d in batch_docs]

    # Embed the batch with retries
    embs = None
    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            if hasattr(embeddings, 'embed_documents'):
                embs = embeddings.embed_documents(texts)
            else:
                embs = [embeddings.embed_query(t) for t in texts]
            break
        except Exception as e:
            wait = 2 ** attempt
            print(f"❌ Embedding batch {start}-{end} failed (attempt {attempt+1}): {e}")
            if attempt < RETRY_ATTEMPTS:
                print(f"   Retrying in {wait}s...")
                time.sleep(wait)
    
    if embs is None:
        print(f"💥 Skipping batch {start}-{end} after failed embedding attempts")
        continue

    # Prepare vectors for upsert
    vectors = []
    for i, emb in enumerate(embs):
        idx = start + i
        doc = batch_docs[i]
        source = doc.metadata.get('source', 'unknown')
        filename = os.path.basename(source) if source != hospital_csv else "hospital_directory.csv"
        
        vectors.append({
            "id": f"doc_{idx}_{hash(filename) % 10000:04d}",
            "values": emb,
            "metadata": {
                "source": source,
                "filename": filename,
                "type": doc.metadata.get('type', 'pdf'),
                "text_preview": (doc.page_content or '')[:300]
            }
        })

    # Upsert with retries
    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            index.upsert(vectors)
            upserted += len(vectors)
            break
        except Exception as e:
            wait = 2 ** attempt
            print(f"❌ Upsert failed for batch {start}-{end} (attempt {attempt+1}): {e}")
            if attempt < RETRY_ATTEMPTS:
                print(f"   Retrying in {wait}s...")
                time.sleep(wait)
    else:
        print(f"💥 Failed to upsert batch {start}-{end} after retries")

    # Progress reporting
    elapsed = time.time() - start_time
    if upserted > 0:
        avg_time = elapsed / upserted
        remaining = total_chunks - upserted
        eta = remaining * avg_time
        
        progress_pct = (upserted / total_chunks) * 100
        print(f"📊 Progress: {upserted}/{total_chunks} ({progress_pct:.1f}%) - ETA: {eta:.1f}s")

print(f"\n🎉 INDEXING COMPLETE!")
print(f"✅ Successfully upserted {upserted} vectors")
print(f"⏱️ Total time: {time.time() - start_time:.1f}s")

# Final summary
print(f"\n📦 INDEX SUMMARY:")
print(f"   - PDF documents: {len(all_pdf_docs)}")
print(f"   - Hospital records: {len(hospital_docs)}")
print(f"   - Total chunks: {len(text_chunks)}")
print(f"   - Vectors in Pinecone: {upserted}")

# Verify index count
try:
    index_stats = index.describe_index_stats()
    print(f"   - Pinecone index total vectors: {index_stats.get('total_vector_count', 'N/A')}")
except Exception as e:
    print(f"   - Could not verify Pinecone stats: {e}")