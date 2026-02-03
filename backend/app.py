import os, time, uuid, json, sqlite3
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, session, flash
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash

from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore as lcp
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import hashlib  

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey_2025"

# -----------------------------
# In-memory session tracking
# -----------------------------
active_sessions = {}   # {session_id: {...}}
client_sessions = {}   # {client_id: [session_ids]}

# -----------------------------
# Load environment keys
# -----------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "keyssss.env"))
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# -----------------------------
# Setup Embeddings + Vector DB
# -----------------------------
embeddings = download_hugging_face_embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-chatbot")
docsearch = lcp(index, embeddings, text_key="text")

# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
PROMPT = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
retriever = docsearch.as_retriever(search_kwargs={"k": 2})

# -----------------------------
# Utility: unique client per user login
# -----------------------------
def get_client_id(request_obj) -> str:
    """Use logged-in username if available; else cookie-based client ID."""
    if "user" in session:
        return f"user_{session['user']}"  # unique chat namespace per login
    
    client_id = request_obj.cookies.get('client_id')
    if not client_id:
        client_id = uuid.uuid4().hex
    return client_id

# -----------------------------
# Conversation memory handling
# -----------------------------
def get_or_create_session_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            "memory": ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            ),
            "messages": [],
            "title": "New Chat",
            "created_at": datetime.utcnow().isoformat(),
            "client_id": None,  # Will be set when session is created
            "username": None    # Will be set when session is created
        }
    return active_sessions[session_id]["memory"]

def create_qa_chain_for_session(session_id: str):
    memory = get_or_create_session_memory(session_id)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda chat_history: chat_history,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        return_generated_question=True
    )

# -----------------------------
# Simplified fallback chain
# -----------------------------
simple_fallback_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are a helpful Indian medical assistant. The user asked: "{question}"

Previous conversation:
{chat_history}

Respond with concise, practical medical information suitable for the question."""
)

def create_fallback_chain_for_session(session_id: str):
    memory = get_or_create_session_memory(session_id)
    return LLMChain(llm=llm, prompt=simple_fallback_prompt, memory=memory, output_key="answer")

# -----------------------------
# Safe File Operations
# -----------------------------
def safe_file_write(file_path, data, max_retries=3):
    """Safely write data to a file with retry logic and atomic operations"""
    for attempt in range(max_retries):
        try:
            # Write to temporary file first
            temp_path = file_path.with_suffix('.tmp')
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            # Replace the original file atomically
            temp_path.replace(file_path)
            return True
            
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.1)
            else:
                print(f"Failed to write {file_path} after {max_retries} attempts")
                return False
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False
    
    return False

def safe_file_delete(file_path, max_retries=3):
    """Safely delete a file with retry logic"""
    for attempt in range(max_retries):
        try:
            file_path.unlink()
            return True
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.2)
            else:
                print(f"Failed to delete {file_path} - file is locked")
                return False
        except FileNotFoundError:
            return True  # File already deleted
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            return False
    
    return False

# -----------------------------
# Session & Chat Management
# -----------------------------
def create_new_session(client_id: str, chat_title: str = None) -> str:
    session_id = uuid.uuid4().hex
    username = session.get("user", "anonymous")
    
    active_sessions[session_id] = {
        "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer"),
        "messages": [],
        "title": chat_title or "New Chat",
        "created_at": datetime.utcnow().isoformat(),
        "client_id": client_id,
        "username": username
    }
    
    if client_id not in client_sessions:
        client_sessions[client_id] = []
    client_sessions[client_id].insert(0, session_id)
    return session_id

def get_all_client_sessions(client_id: str):
    sessions = []
    current_username = session.get("user", "anonymous")
    seen = set()  # Track unique session IDs to prevent duplicates
    
    # Load from active sessions (memory)
    if client_id in client_sessions:
        for session_id in client_sessions[client_id]:
            if session_id in active_sessions:
                s = active_sessions[session_id]
                # Only include sessions that belong to the current user
                if s.get("username") == current_username and session_id not in seen:
                    sessions.append({
                        "id": session_id,
                        "title": s["title"],
                        "created_at": s["created_at"],
                        "message_count": len(s["messages"])
                    })
                    seen.add(session_id)
    
    # Also load from disk for persistence
    conversations_dir = Path("conversations")
    if conversations_dir.exists():
        for json_file in conversations_dir.glob("*.json"):
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    session_id = session_data.get("id")
                    # Check if this session belongs to the current user and isn't already added
                    if (session_data.get("username") == current_username and 
                        session_data.get("client_id") == client_id and 
                        session_id not in seen):
                        sessions.append({
                            "id": session_id,
                            "title": session_data["title"],
                            "created_at": session_data["created_at"],
                            "message_count": len(session_data.get("messages", []))
                        })
                        seen.add(session_id)
            except Exception as e:
                print(f"Error loading session from {json_file}: {e}")
    
    # Sort by creation date, newest first
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    return sessions

def save_session_to_disk(session_id: str):
    if session_id not in active_sessions: 
        return
        
    s = active_sessions[session_id]
    data = {
        "id": session_id, 
        "title": s["title"], 
        "created_at": s["created_at"], 
        "messages": s["messages"],
        "client_id": s["client_id"],
        "username": s["username"]
    }
    
    path = Path("conversations") / f"{session_id}.json"
    path.parent.mkdir(exist_ok=True)
    
    if not safe_file_write(path, data):
        print(f"Warning: Could not save session {session_id} to disk")

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    username = session["user"]
    return render_template("chat.html", username=username)

@app.route("/new_chat", methods=["POST"])
def new_chat():
    data = request.get_json() or {}
    chat_title = data.get('title', 'New Chat')
    client_id = get_client_id(request)
    session_id = create_new_session(client_id, chat_title)
    resp = make_response(jsonify({"success": True, "session_id": session_id, "title": chat_title}))
    resp.set_cookie('client_id', client_id)
    return resp

@app.route("/get_client_sessions", methods=["GET"])
def get_client_sessions_route():
    client_id = get_client_id(request)
    sessions = get_all_client_sessions(client_id)
    return jsonify({"sessions": sessions})

@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("msg")
    session_id = data.get("session_id")
    client_id = get_client_id(request)
    
    # Your existing session setup code...
    
    active_sessions[session_id]["messages"].append({"from": "user", "text": msg})

    try:
        # DEDUPLICATION: Get source docs and remove duplicates
        source_docs = retriever.get_relevant_documents(msg)
        
        # Remove duplicates based on content similarity
        unique_docs = []
        seen_content = set()
        
        for doc in source_docs:
            # Create a content signature (first 200 chars + source)
            content_preview = doc.page_content[:200].lower().strip()
            source_file = doc.metadata.get('source', '')
            content_signature = f"{source_file}_{hashlib.md5(content_preview.encode()).hexdigest()[:10]}"
            
            if content_signature not in seen_content:
                seen_content.add(content_signature)
                unique_docs.append(doc)
        
        print(f"🔍 Deduplication: {len(source_docs)} → {len(unique_docs)} documents")
        
        # Create context from unique docs only
        context = "\n\n".join([doc.page_content for doc in unique_docs])
        
        qa_chain = create_qa_chain_for_session(session_id)
        chat_history = "\n".join([f"{m['from']}: {m['text']}" for m in active_sessions[session_id]["messages"][-5:]])
        
        result = qa_chain({
            "question": msg, 
            "chat_history": chat_history,
            "context": context  # Pass deduplicated context
        })
        answer = result["answer"]
        
    except Exception as e:
        print("Error:", e)
        # Fallback with deduplication too
        fallback = create_fallback_chain_for_session(session_id)
        result = fallback({"question": msg, "chat_history": ""})
        answer = result["answer"]

    active_sessions[session_id]["messages"].append({"from": "bot", "text": answer})
    save_session_to_disk(session_id)
    
    return jsonify({"answer": answer, "session_id": session_id})

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy", 
        "active_sessions": len(active_sessions),
        "total_clients": len(client_sessions),
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "llm": "available" if llm else "unavailable",
            "vector_store": "connected" if docsearch else "unavailable"
        }
    })

# -----------------------------
# Auth System (Login/Signup/Logout)
# -----------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit(); conn.close()
init_db()

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            session["user"] = username
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password", "danger")

    return render_template("auth.html", mode="login")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm = request.form.get("confirm")
        if password != confirm:
            flash("Passwords do not match", "warning")
            return render_template("auth.html", mode="signup")

        conn = sqlite3.connect("users.db"); cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                        (username, generate_password_hash(password)))
            conn.commit()
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
        finally:
            conn.close()
    return render_template("auth.html", mode="signup")

@app.route("/api/chats/<session_id>", methods=["GET"])
def get_chat_session(session_id):
    current_username = session.get("user", "anonymous")
    client_id = get_client_id(request)
    
    # SECURITY CHECK: First check active sessions
    if session_id in active_sessions:
        session_data = active_sessions[session_id]
        # Verify ownership
        if (session_data.get("username") == current_username and 
            session_data.get("client_id") == client_id):
            return jsonify({
                "id": session_id,
                "title": session_data["title"],
                "created_at": session_data["created_at"],
                "messages": session_data["messages"]
            })
        else:
            return jsonify({"error": "Access denied"}), 403
    
    # Try to load from disk with security check
    path = Path("conversations") / f"{session_id}.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            session_data = json.load(f)
            # Verify ownership
            if (session_data.get("username") == current_username and 
                session_data.get("client_id") == client_id):
                return jsonify(session_data)
            else:
                return jsonify({"error": "Access denied"}), 403
    
    return jsonify({"error": "Session not found"}), 404

@app.route("/delete_chat/<session_id>", methods=["DELETE"])
def delete_chat(session_id):
    try:
        current_username = session.get("user", "anonymous")
        client_id = get_client_id(request)
        
        print(f"Delete request - session_id: {session_id}, username: {current_username}, client_id: {client_id}")
        
        # SECURITY CHECK: Verify ownership before deletion
        can_delete = False
        
        # Check active sessions
        if session_id in active_sessions:
            session_data = active_sessions[session_id]
            if (session_data.get("username") == current_username and 
                session_data.get("client_id") == client_id):
                can_delete = True
                del active_sessions[session_id]
                print(f"Deleted from active sessions: {session_id}")
        
        # Check disk sessions with retry logic
        path = Path("conversations") / f"{session_id}.json"
        if path.exists():
            # Retry logic for file deletion
            for attempt in range(3):
                try:
                    with path.open("r", encoding="utf-8") as f:
                        session_data = json.load(f)
                    
                    if (session_data.get("username") == current_username and 
                        session_data.get("client_id") == client_id):
                        can_delete = True
                        if safe_file_delete(path):
                            print(f"Deleted from disk: {session_id}")
                        else:
                            print(f"File deletion failed but continuing: {session_id}")
                        break
                        
                except PermissionError:
                    if attempt < 2:  # First two attempts
                        print(f"File locked, retrying deletion in 0.2s... (attempt {attempt + 1})")
                        time.sleep(0.2)
                    else:
                        print(f"Could not delete file {session_id}.json - file is locked")
                        # Still consider it a success if we removed from active sessions
                        if can_delete:
                            return jsonify({"success": True, "warning": "File could not be deleted due to locking, but removed from active sessions"})
                except Exception as e:
                    print(f"Error reading/deleting file: {e}")
                    break
        
        # Remove from client sessions
        for cid, sessions in client_sessions.items():
            if session_id in sessions:
                sessions.remove(session_id)
                print(f"Removed from client_sessions for client {cid}")
        
        if can_delete:
            return jsonify({"success": True, "message": "Chat deleted successfully"})
        else:
            print("Access denied - ownership verification failed")
            return jsonify({"success": False, "error": "Access denied or chat not found"}), 403
            
    except Exception as e:
        print(f"Error deleting chat: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))
@app.route("/delete_account", methods=["POST"])
def delete_account():
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 403

    username = session["user"]

    try:
        # Delete user from DB
        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        conn.close()

        # Remove conversations from disk
        conversations_dir = Path("conversations")
        if conversations_dir.exists():
            for f in conversations_dir.glob("*.json"):
                try:
                    with f.open("r", encoding="utf-8") as file:
                        data = json.load(file)
                        if data.get("username") == username:
                            safe_file_delete(f)
                except:
                    pass

        # Remove from active_sessions
        to_delete = []
        for sid, sdata in active_sessions.items():
            if sdata.get("username") == username:
                to_delete.append(sid)
        for sid in to_delete:
            del active_sessions[sid]

        # Remove from client_sessions
        client_id = f"user_{username}"
        if client_id in client_sessions:
            del client_sessions[client_id]

        # Logout user
        session.pop("user", None)

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)