<div align="center">

# 🩺 CareBot — Clinical RAG Assistant

**Evidence-Grounded Medical Decision-Support System powered by FastAPI, BM25 Probabilistic Retrieval, and Google Gemini 2.5.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LLM](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Deployment](https://img.shields.io/badge/Deployed-Vercel-black.svg)](https://vercel.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Overview

**CareBot** is a production-grade, evidence-grounded Retrieval-Augmented Generation (RAG) assistant designed for medical inquiry, clinical question-answering, and patient document review. 

To eliminate clinical hallucination risks, CareBot strictly grounds its responses on verified medical literature and user-uploaded clinical documentation, providing transparent citations with exact source names and page numbers for every answer.

---

## ✨ Key Features

- **Evidence-Grounded RAG Engine**: Utilizes probabilistic **BM25 / TF-IDF** scoring with document length normalization and stop-word filtering over 2,200+ indexed clinical sections.
- **Dual-Tier Knowledge Grounding**:
  - **Base Encyclopedia**: 759 pages from the *Gale Encyclopedia of Medicine*, pre-indexed into a fast gzipped cache for sub-second retrieval without cold starts.
  - **Dynamic Patient Document Ingestion**: Ingest and index arbitrary clinical records (PDF/TXT) on the fly.
- **Three Query Modes**:
  1. 🔀 **Combined Knowledge**: Synthesizes evidence across both base reference literature and uploaded clinical records.
  2. 📄 **Uploaded Docs Only**: Restricts retrieval strictly to user-uploaded consultation notes or lab reports.
  3. 📚 **Base Encyclopedia Only**: Queries validated reference encyclopedias exclusively.
- **Frontier LLM Integration**: Powered by Google's `gemini-2.5-flash` with automatic failover to `gemini-flash-latest` and Groq (`llama-3.3-70b-versatile`).
- **Hands-Free Clinical Voice Dictation**: Integrated Web Speech API for voice queries.
- **Interactive Fluid Motion UX**: Sleek clinical dark-theme with 60 FPS fluid-motion particle cursor flow engine.
- **Exportable Records**: One-click JSON export of clinical consultation histories.

---

## 🏗️ Architecture & Project Structure

The project has been cleaned and organized into a clean production architecture:

```
RAG-based-medical-chatbot/
├── api/
│   ├── index.py          # FastAPI application & complete RAG engine
│   └── index.html        # Interactive clinical web consultation UI
├── data/
│   └── gale_knowledge.json.gz  # Fast pre-extracted 759-page clinical knowledge base
├── public/
│   └── index.html        # Public frontend assets
├── .env.example          # Template for environment variables
├── .gitignore            # Clean git configuration
├── app.py                # Local development entrypoint
├── index.html            # Root frontend template
├── requirements.txt      # Minimal production Python dependencies
├── vercel.json           # Vercel serverless deployment config
└── README.md             # Project documentation
```

---

## 🚀 Quick Start (Local Development)

### 1. Clone the repository
```bash
git clone https://github.com/nazish-saghir-ahmed/RAG-based-medical-chatbot.git
cd RAG-based-medical-chatbot
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/):
```env
GEMINI_API_KEY="AIzaSy..."
```

### 5. Launch the application
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:8000
```

---

## ☁️ Deployment on Vercel

CareBot is configured for zero-configuration serverless deployment on Vercel:

1. Import your GitHub repository into [Vercel](https://vercel.com/).
2. Navigate to **Project Settings** $\rightarrow$ **Environment Variables**.
3. Add:
   * **Key**: `GEMINI_API_KEY`
   * **Value**: Your Google Gemini API key
   * **Target Environments**: Production, Preview, Development
4. Deploy! Vercel handles serverless routing through `vercel.json` and `api/index.py`.

---

## 🧪 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive CareBot consultation UI |
| `GET` | `/api/health` | Health-check endpoint reporting chunk index counts & key status |
| `POST` | `/api/chat` | Main RAG inquiry endpoint (accepts query, mode, history) |
| `POST` | `/api/upload` | Multipart endpoint for parsing & indexing patient PDFs/TXTs |
| `POST` | `/api/clear_uploads` | Clears all session-uploaded patient records |

---

## 🛡️ Medical Disclaimer

*CareBot is an educational and clinical demonstration tool. It provides evidence-grounded information based on provided reference material and is not a substitute for professional medical judgment, diagnosis, or treatment.*
