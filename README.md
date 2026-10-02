# CareBot 🩺 - Medical RAG AI Assistant

CareBot is an intelligent, Retrieval-Augmented Generation (RAG) powered conversational medical assistant built with **Streamlit**, **LangChain**, **HuggingFace Embeddings**, and **Google Gemini / Groq / OpenAI LLMs**.

CareBot retrieves accurate, context-grounded information from clinical medical texts and encyclopedias, strictly preventing hallucinated responses and citing every source document and page number.

---

## 🌟 Key Features

- 📚 **Dual-Source Retrieval-Augmented Generation (RAG)**:
  - **Base Knowledge Base**: Pre-embedded medical encyclopedia (`ChromaDB` / `FAISS`).
  - **Dynamic Document Upload**: On-the-fly embedding and querying of custom PDF and TXT medical documents.
- 🔀 **Flexible Query Modes**:
  - *Combined Mode* (default): Retrieves context from both uploaded documents and the core medical knowledge base.
  - *Uploaded Documents Only*: Focuses strictly on the user's uploaded files.
  - *Base Knowledge Only*: Queries standard medical references.
- 💬 **Session Chat Management**:
  - Full conversation history with clean dark-mode UI.
  - Instant chat export / download as JSON with timestamps and citations.
  - One-click conversation reset.
- 🎯 **Strict Context-Aware Answers**: Prompt templates enforce factual generation strictly grounded in retrieved passages with source citations.
- 🎙️ **Voice Recognition Ready**: Built-in support for speech recognition input.

---

## 🏗️ Architecture

```
User Query / Document Upload
            │
    ┌───────┴────────────────────────┐
    ▼                                ▼
[Uploaded PDF/TXT]          [Base Knowledge Base]
(Dynamic Embeddings)           (Pre-computed ChromaDB)
    │                                │
    └───────┬────────────────────────┘
            ▼
   [Sentence-Transformers] (all-MiniLM-L6-v2)
            ▼
    [Vector Retriever] (Cosine Similarity, Top-k)
            ▼
    [Prompt Template + Context]
            ▼
     [LLM Generation] (Gemini / Groq / OpenAI)
            ▼
  Answer + Source Citations
```

---

## 📂 Project Structure

```
PROJECT_RAG/
├── data/                               # Reference medical source documents
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
├── streamlit_app.py                    # Main CareBot Streamlit application
├── streamlit_app_explain.py            # Annotated version with step-by-step documentation
├── connectmemorywithllm.py             # Backend RAG retrieval chain & CLI tester
├── Creatememoryforllm.py               # Vectorstore index generation script (FAISS)
├── carebot.py                          # Groq-based prototype implementation
├── CHATBOT_FEATURES.md                 # Detailed feature specifications
├── concepts.md                         # Comprehensive RAG concepts & architectural guide
├── requirements.txt                    # Project dependencies
├── .env.example                        # Environment variables template
└── .gitignore                          # Git ignore rules for virtualenvs, caches & keys
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd PROJECT_RAG
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```

Edit `.env` and supply:
```env
GEMINI_API_KEY="your_google_gemini_api_key_here"
# Optional alternatives:
GROQ_API_KEY="your_groq_api_key_here"
LONGCAT_API_KEY="your_longcat_api_key_here"
HF_TOKEN="your_huggingface_token_if_needed"
```

### 5. Run the Application
Launch the Streamlit web application:
```bash
streamlit run streamlit_app.py
```

Or test the RAG chain in CLI mode:
```bash
python connectmemorywithllm.py
```

---

## 🛡️ Privacy & Safety Disclaimer

CareBot is an AI assistant designed for educational and informational reference only. It should not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for clinical decisions.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
