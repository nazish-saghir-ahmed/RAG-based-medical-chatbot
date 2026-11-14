import os
import io
import json
import gzip
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

STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'could',
    'did', 'do', 'does', 'doing', 'don', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has',
    'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into',
    'is', 'isn', 'it', 'its', 'itself', 'let', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not',
    'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
    'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn',
    'we', 'were', 'weren', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'with', 'won', 'would',
    'you', 'your', 'yours', 'yourself', 'yourselves'
}

# In-memory store for session documents
uploaded_docs_store: List[Dict[str, Any]] = []
base_docs_store: List[Dict[str, Any]] = []
base_doc_frequencies: Dict[str, int] = {}

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', text)]

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks

def build_chunk_metadata(text: str, source: str, page: int) -> Dict[str, Any]:
    tokens = tokenize(text)
    meaningful = [t for t in tokens if t not in STOP_WORDS]
    counts: Dict[str, int] = {}
    for t in meaningful:
        counts[t] = counts.get(t, 0) + 1
    return {
        "source": source,
        "page": page,
        "text": text.strip(),
        "tokens": set(tokens),
        "meaningful_tokens": set(meaningful),
        "meaningful_counts": counts,
        "token_count": len(tokens)
    }

def extract_pdf_chunks(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    chunks = []
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                continue
            for chunk in chunk_text(text, chunk_size=300, overlap=40):
                meta = build_chunk_metadata(chunk, filename, page_idx + 1)
                if meta["meaningful_tokens"]:
                    chunks.append(meta)
    except Exception as e:
        print(f"Error parsing PDF {filename}: {e}")
    return chunks

def extract_txt_chunks(text: str, filename: str) -> List[Dict[str, Any]]:
    chunks = []
    text = re.sub(r'\s+', ' ', text).strip()
    for chunk in chunk_text(text, chunk_size=300, overlap=40):
        meta = build_chunk_metadata(chunk, filename, 1)
        if meta["meaningful_tokens"]:
            chunks.append(meta)
    return chunks

def load_base_knowledge():
    global base_docs_store, base_doc_frequencies
    if base_docs_store:
        return

    # 1. Try precomputed gzipped JSON (instant load for 759 pages, zero CPU overhead)
    precomputed_paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "gale_knowledge.json.gz"),
        os.path.join(os.path.dirname(__file__), "data", "gale_knowledge.json.gz"),
        os.path.join("data", "gale_knowledge.json.gz"),
    ]

    loaded_from_gz = False
    for gz_path in precomputed_paths:
        if os.path.exists(gz_path):
            try:
                with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                    pages = json.load(f)
                for p in pages:
                    for ch in chunk_text(p["text"]):
                        meta = build_chunk_metadata(ch, "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf", p["page"])
                        if meta["meaningful_tokens"]:
                            base_docs_store.append(meta)
                loaded_from_gz = True
                break
            except Exception as e:
                print(f"Error loading {gz_path}: {e}")

    # 2. Fallback to raw text files
    if not loaded_from_gz:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        if not os.path.exists(data_dir):
            data_dir = "data"
        if os.path.exists(data_dir):
            for fname in os.listdir(data_dir):
                fpath = os.path.join(data_dir, fname)
                if fname.lower().endswith(".txt") and os.path.isfile(fpath):
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            base_docs_store.extend(extract_txt_chunks(f.read(), fname))
                    except Exception as e:
                        print(f"Error reading base document {fname}: {e}")

    # Compute global document frequencies for BM25 term weighting
    df: Dict[str, int] = {}
    for doc in base_docs_store:
        for t in doc.get("meaningful_tokens", set()):
            df[t] = df.get(t, 0) + 1
    base_doc_frequencies = df

def retrieve_relevant_chunks(query: str, doc_pool: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    if not doc_pool:
        return []

    q_tokens = [w for w in tokenize(query) if w not in STOP_WORDS]
    if not q_tokens:
        q_tokens = tokenize(query)
    if not q_tokens:
        return []

    N_ref = max(100, len(base_docs_store))
    scored = []
    query_lower = query.lower()

    for doc in doc_pool:
        score = 0.0
        doc_tokens = doc.get("tokens", set())
        meaningful_counts = doc.get("meaningful_counts", {})
        token_count = doc.get("token_count", 100)

        for qt in q_tokens:
            if qt in meaningful_counts:
                tf = meaningful_counts[qt]
                doc_freq = base_doc_frequencies.get(qt, 1)
                # BM25 probabilistic IDF: log(1 + (N - df + 0.5) / (df + 0.5))
                idf = math.log(1.0 + (N_ref - min(doc_freq, N_ref - 1) + 0.5) / (doc_freq + 0.5))
                tf_norm = (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (token_count / 300.0)))
                score += idf * tf_norm
            elif qt in doc_tokens:
                score += 0.5

        if score > 0:
            # Boost chunks containing the exact multi-word search phrase
            if query_lower in doc["text"].lower():
                score *= 2.5
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

def get_api_key(name: str) -> Optional[str]:
    # Check exact name, lower case, upper case, and common aliases
    aliases = [name, name.upper(), name.lower()]
    if "gemini" in name.lower():
        aliases.extend(["GOOGLE_API_KEY", "GEMINI_KEY", "AI_API_KEY", "gemini_key", "google_api_key"])
    if "groq" in name.lower():
        aliases.extend(["GROQ_KEY", "groq_key"])

    for alias in aliases:
        val = os.getenv(alias)
        if val and val.strip():
            return val.strip()
    return None

def generate_llm_response(prompt: str, context: str) -> str:
    gemini_key = get_api_key("GEMINI_API_KEY")
    groq_key = get_api_key("GROQ_API_KEY")

    system_instruction = (
        "You are CareBot, an expert, empathetic, and evidence-based AI medical assistant. "
        "Use the provided context to answer the user's question accurately. "
        "If the answer cannot be determined from the context, state clearly what is known and advise consulting a healthcare professional. "
        "Do not invent medical facts. Provide clear, clinical, yet accessible answers."
    )

    full_prompt = f"{system_instruction}\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"

    # 1. Google Gemini via REST (models/gemini-2.5-flash with fallback to gemini-flash-latest)
    if gemini_key:
        try:
            import requests
            try:
                import certifi
                verify_ssl = certifi.where()
            except ImportError:
                verify_ssl = True

            for model_name in ["gemini-2.5-flash", "gemini-flash-latest"]:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
                payload = {
                    "contents": [{
                        "parts": [{"text": full_prompt}]
                    }]
                }
                try:
                    res = requests.post(url, json=payload, timeout=25, verify=verify_ssl)
                except Exception:
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
                model='gemini-2.5-flash',
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
    load_base_knowledge()
    gemini_key = get_api_key("GEMINI_API_KEY")
    groq_key = get_api_key("GROQ_API_KEY")
    # Safe preview: reveal first 4 characters and length to verify exact loaded key without exposing secrets
    preview = f"{gemini_key[:4]}...({len(gemini_key)} chars)" if gemini_key else None
    return {
        "status": "online",
        "service": "CareBot RAG API",
        "uploaded_chunks": len(uploaded_docs_store),
        "base_chunks": len(base_docs_store),
        "has_gemini_key": bool(gemini_key),
        "gemini_key_preview": preview,
        "has_groq_key": bool(groq_key),
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
