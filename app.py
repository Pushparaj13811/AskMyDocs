import os
import uuid
import shutil
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
import httpx
import uvicorn
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
load_dotenv()

# --- CONSTANTS ---
UPLOAD_DIR = Path("uploads")
DB_DIR = Path("chroma_db")
TEMPLATE_DIR = Path("templates")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please create a .env file and add it.")

# --- INITIALIZATION ---
app = FastAPI()

# Create necessary directories
UPLOAD_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR.mkdir(exist_ok=True)

# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Setup AI and DB models
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel('gemini-1.5-flash')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

# In-memory storage for session data and document collections
sessions: Dict[str, List[Dict[str, str]]] = {}
document_collections: Dict[str, chromadb.Collection] = {}

# --- HELPER FUNCTIONS ---

def extract_text_from_pdf(file_path: Path) -> str:
    """Extracts text from a PDF file."""
    try:
        with fitz.open(file_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF file: {e}")

def extract_text_from_txt(file_path: Path) -> str:
    """Extracts text from a TXT file."""
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing TXT file: {e}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Splits text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def get_or_create_collection(name: str) -> chromadb.Collection:
    """Gets or creates a ChromaDB collection."""
    if name in document_collections:
        return document_collections[name]
    
    collection = chroma_client.get_or_create_collection(name=name)
    document_collections[name] = collection
    return collection

def process_and_store_document(file_path: Path, collection_name: str):
    """Processes an uploaded document and stores it in ChromaDB."""
    if file_path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif file_path.suffix.lower() == ".txt":
        text = extract_text_from_txt(file_path)
    else:
        return  # Or raise an error for unsupported file types

    text_chunks = chunk_text(text)
    if not text_chunks:
        return

    collection = get_or_create_collection(collection_name)
    
    # Store chunks in ChromaDB
    embeddings = embedding_model.encode(text_chunks, show_progress_bar=False).tolist()
    ids = [str(uuid.uuid4()) for _ in text_chunks]
    
    collection.add(
        embeddings=embeddings,
        documents=text_chunks,
        ids=ids
    )

def get_formatted_memory(session_id: str) -> str:
    """Formats chat history for the AI prompt."""
    memory = sessions.get(session_id, [])
    if not memory:
        return "No previous conversation history."
    
    return "\n".join([f"Human: {item['human']}\nAI: {item['ai']}" for item in memory])

# --- FastAPI Endpoints ---

@app.on_event("startup")
def startup_event():
    """Load existing ChromaDB collections on startup."""
    for collection in chroma_client.list_collections():
        document_collections[collection.name] = collection

@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    """Renders the main chat interface."""
    session_id = request.cookies.get("session_id") or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    response = templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_history": sessions[session_id],
            "documents": list(document_collections.keys()),
        },
    )
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return response

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Handles document uploads, processing, and storage."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        collection_name = file.filename.rsplit(".", 1)[0].replace(" ", "_")
        process_and_store_document(file_path, collection_name)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"An error occurred: {e}"})


    return JSONResponse(content={"filename": file.filename, "collection_name": collection_name})

@app.get("/documents", response_class=JSONResponse)
async def get_documents():
    """Returns a list of available documents."""
    return {"documents": list(document_collections.keys())}

@app.post("/chat")
async def handle_chat(
    request: Request,
    question: str = Form(...),
    document_names: List[str] = Form(...),
):
    """Handles user chat messages, retrieves context from multiple documents, and gets an AI response."""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session.")

    if not document_names:
        raise HTTPException(status_code=400, detail="Please select at least one document to chat with.")

    # 1. Embed the user's question
    question_embedding = embedding_model.encode([question]).tolist()
    
    # 2. Query ChromaDB for relevant context from all selected documents
    all_context_chunks = []
    for doc_name in document_names:
        if doc_name not in document_collections:
            continue # Skip if a document is not found, or handle error as needed
        
        collection = document_collections[doc_name]
        results = collection.query(query_embeddings=question_embedding, n_results=3) # Get top 3 chunks per doc
        
        if results and results["documents"] and results["documents"][0]:
            all_context_chunks.extend(results["documents"][0])

    # Remove duplicates and combine context
    context = "\n---\n".join(list(dict.fromkeys(all_context_chunks)))
    
    # 3. Get formatted chat history (memory)
    memory = get_formatted_memory(session_id)

    # 4. Construct the prompt for the LLM
    prompt = f"""You are an experienced professor conducting an AI workshop.

    Your goal is to answer questions in a clear, thoughtful, and educational manner, using the provided context and conversation history. Speak with the depth, precision, and tone of an expert educator. Encourage understanding, and if applicable, relate concepts to foundational ideas.

    If the answer is not present in the context, say so respectfully and avoid guessing.

    ---

    🧠 Conversation History:
    {memory}

    📄 Document Context (from: {", ".join(document_names)}):
    {context}

    ❓ Student's Question:
    {question}

    ---

    🎓 Please provide a well-structured, explanatory answer as if you are guiding a student. Clarify key concepts and their significance. Use examples when helpful. Aim to deepen the student's understanding with your response.
    """


    # 5. Get the AI's response
    try:
        response = llm.generate_content(prompt)
        ai_answer = response.text
    except Exception as e:
        ai_answer = f"Error generating response from AI: {e}"

    # 6. Update session memory
    sessions[session_id].append({"human": question, "ai": ai_answer})

    return JSONResponse(content={"human": question, "ai": ai_answer})


if __name__ == "__main__":
    # Check for the template file before starting the server
    if not (TEMPLATE_DIR / "index.html").exists():
        print("Error: 'templates/index.html' not found.")
        print("Please create the index.html file in the 'templates' directory.")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000) 