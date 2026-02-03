from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
import glob
import logging
from pypdf.errors import PdfReadError



#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents


def load_pdf_safe(directory: str):
    """Load PDFs from a directory one-by-one, skipping files that fail to parse.

    Returns a list of langchain Document objects.
    """
    docs = []
    pdf_paths = sorted(glob.glob(os.path.join(directory, "*.pdf")))

    for path in pdf_paths:
        filename = os.path.basename(path)
        # Skip temporary Office files or files that start with ~$
        if filename.startswith('~$'):
            logging.info(f"Skipping temporary file: {filename}")
            continue

        try:
            loader = PyPDFLoader(path)
            subdocs = loader.load()
            docs.extend(subdocs)
            logging.info(f"Loaded PDF: {filename} ({len(subdocs)} pages/parts)")
        except (PdfReadError, Exception) as e:
            logging.warning(f"Error loading file {filename}: {e} — skipping")
            continue

    return docs



#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings