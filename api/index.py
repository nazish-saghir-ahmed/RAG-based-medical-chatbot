import os
import io
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pypdf

load_dotenv()

app = FastAPI(title="CareBot RAG Medical Assistant", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for session documents
uploaded_docs_store: List[Dict[str, Any]] = []
base_docs_store: List[Dict[str, Any]] = []

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', text)]

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks

def extract_pdf_chunks(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    chunks = []
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            for chunk in chunk_text(text, chunk_size=300, overlap=40):
                tokens = tokenize(chunk)
                if tokens:
                    chunks.append({
                        "source": filename,
                        "page": page_idx + 1,
                        "text": chunk.strip(),
                        "tokens": set(tokens),
                        "token_count": len(tokens)
                    })
    except Exception as e:
        print(f"Error parsing PDF {filename}: {e}")
    return chunks

def extract_txt_chunks(text: str, filename: str) -> List[Dict[str, Any]]:
    chunks = []
    for chunk in chunk_text(text, chunk_size=300, overlap=40):
        tokens = tokenize(chunk)
        if tokens:
            chunks.append({
                "source": filename,
                "page": 1,
                "text": chunk.strip(),
                "tokens": set(tokens),
                "token_count": len(tokens)
            })
    return chunks

def load_base_knowledge():
    global base_docs_store
    if base_docs_store:
        return
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    if not os.path.exists(data_dir):
        data_dir = "data"
    
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            fpath = os.path.join(data_dir, fname)
            if fname.lower().endswith(".pdf") and os.path.isfile(fpath):
                try:
                    with open(fpath, "rb") as f:
                        base_docs_store.extend(extract_pdf_chunks(f.read(), fname))
                except Exception as e:
                    print(f"Error reading base document {fname}: {e}")
            elif fname.lower().endswith(".txt") and os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        base_docs_store.extend(extract_txt_chunks(f.read(), fname))
                except Exception as e:
                    print(f"Error reading base document {fname}: {e}")

def retrieve_relevant_chunks(query: str, doc_pool: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    query_tokens = set(tokenize(query))
    if not query_tokens or not doc_pool:
        return []

    scored = []
    for doc in doc_pool:
        common = query_tokens.intersection(doc["tokens"])
        if common:
            score = len(common) / (math.sqrt(doc["token_count"]) + 1)
            if query.lower() in doc["text"].lower():
                score *= 2.0
            scored.append((score, doc))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

def generate_llm_response(prompt: str, context: str) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    system_instruction = (
        "You are CareBot, an expert, empathetic, and evidence-based AI medical assistant. "
        "Use the provided context to answer the user's question accurately. "
        "If the answer cannot be determined from the context, state clearly what is known and advise consulting a healthcare professional. "
        "Do not invent medical facts. Provide clear, clinical, yet accessible answers."
    )
    
    full_prompt = f"{system_instruction}\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"

    # 1. Google Gemini
    if gemini_key:
        try:
            import requests
            try:
                import certifi
                verify_ssl = certifi.where()
            except ImportError:
                verify_ssl = True

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [{
                    "parts": [{"text": full_prompt}]
                }]
            }
            try:
                res = requests.post(url, json=payload, timeout=25, verify=verify_ssl)
            except requests.exceptions.SSLError:
                res = requests.post(url, json=payload, timeout=25, verify=False)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
        except Exception as e_rest:
            print(f"Gemini REST error: {e_rest}")
            
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=full_prompt,
            )
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Gemini SDK error: {e}")
            
    # 2. Groq
    if groq_key:
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
                ],
                "temperature": 0.2,
                "max_tokens": 1024,
            }
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Groq API error: {e}")

    # Fallback if no API key configured
    return (
        f"Based on retrieved medical reference context:\n\n"
        f"{context[:600]}...\n\n"
        f"*(Note: To enable live LLM responses on Vercel, set `GEMINI_API_KEY` or `GROQ_API_KEY` in your Vercel Project Settings > Environment Variables).* "
    )

class ChatRequest(BaseModel):
    query: str
    mode: str = "combined"
    history: Optional[List[Dict[str, Any]]] = []

@app.on_event("startup")
def startup_event():
    load_base_knowledge()

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "CareBot RAG API",
        "uploaded_chunks": len(uploaded_docs_store),
        "base_chunks": len(base_docs_store),
        "has_gemini_key": bool(os.getenv("GEMINI_API_KEY")),
        "has_groq_key": bool(os.getenv("GROQ_API_KEY")),
    }

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    global uploaded_docs_store
    added_chunks = 0
    uploaded_files_summary = []
    
    for file in files:
        contents = await file.read()
        filename = file.filename
        if filename.lower().endswith(".pdf"):
            chunks = extract_pdf_chunks(contents, filename)
            uploaded_docs_store.extend(chunks)
            added_chunks += len(chunks)
            uploaded_files_summary.append({"name": filename, "chunks": len(chunks), "type": "PDF"})
        elif filename.lower().endswith(".txt"):
            text = contents.decode("utf-8", errors="ignore")
            chunks = extract_txt_chunks(text, filename)
            uploaded_docs_store.extend(chunks)
            added_chunks += len(chunks)
            uploaded_files_summary.append({"name": filename, "chunks": len(chunks), "type": "TXT"})

    return {
        "message": f"Successfully processed {len(files)} document(s). Added {added_chunks} indexable chunks.",
        "files": uploaded_files_summary,
        "total_uploaded_chunks": len(uploaded_docs_store)
    }

@app.post("/api/clear_uploads")
def clear_uploads():
    global uploaded_docs_store
    uploaded_docs_store.clear()
    return {"message": "Uploaded documents cleared", "total_uploaded_chunks": 0}

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    load_base_knowledge()
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    mode = req.mode.lower()
    doc_pool = []
    
    if mode == "uploaded":
        doc_pool = uploaded_docs_store
    elif mode == "base":
        doc_pool = base_docs_store
    else:  # combined
        doc_pool = uploaded_docs_store + base_docs_store

    relevant_chunks = retrieve_relevant_chunks(query, doc_pool, top_k=4)

    if not relevant_chunks:
        context = "No specific reference documents found in selected knowledge store."
        sources = []
    else:
        context = "\n\n---\n\n".join(
            [f"[Source: {c['source']} (Page {c['page']})]\n{c['text']}" for c in relevant_chunks]
        )
        sources = [
            {
                "source": c["source"],
                "page": c["page"],
                "snippet": c["text"][:180] + "..." if len(c["text"]) > 180 else c["text"]
            }
            for c in relevant_chunks
        ]

    answer = generate_llm_response(query, context)

    return {
        "query": query,
        "mode": mode,
        "answer": answer,
        "sources": sources,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/", response_class=HTMLResponse)
def index_page():
    html_paths = [
        os.path.join(os.path.dirname(__file__), "index.html"),
        os.path.join(os.path.dirname(__file__), "..", "public", "index.html"),
        os.path.join(os.path.dirname(__file__), "..", "index.html"),
        os.path.join("public", "index.html"),
        os.path.join("api", "index.html"),
        "index.html",
    ]
    for path in html_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
            except Exception:
                pass
    return HTMLResponse(content="<h1>CareBot UI</h1><p>Could not load index.html template.</p>")
