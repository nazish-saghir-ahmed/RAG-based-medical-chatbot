# ── IMPORTS ───────────────────────────────────────────────────────────────────

import os                        # Access environment variables and file paths
from pathlib import Path         # Handle file/folder paths in a clean way
from dotenv import load_dotenv   # Load secret keys from .env file
import streamlit as st           # The framework that builds the entire web UI
import json                      # Convert Python data to/from JSON format
import tempfile                  # Create temporary folders to store uploaded files
from datetime import datetime    # Get current date/time (used in export filename)
import io                        # Lets us treat raw bytes as a file (used in audio)

# Try to import speech recognition — if not installed, just disable mic feature
try:
    import speech_recognition as sr   # Google Speech-to-Text library
    HAS_SPEECH_RECOGNITION = True     # Flag: mic feature is available
except ImportError:
    HAS_SPEECH_RECOGNITION = False    # Flag: mic feature disabled silently

# LangChain + AI imports
from langchain_chroma import Chroma                                      # Vector database to store/search document embeddings
from langchain_classic.chains import RetrievalQA                        # The chain that connects retriever + LLM + prompt together
from langchain_huggingface import HuggingFaceEmbeddings                 # Converts text into numeric vectors (embeddings)
from langchain_community.document_loaders import TextLoader, PyPDFLoader # Load .txt and .pdf files into LangChain documents
from langchain_core.prompts import PromptTemplate                        # Build a reusable prompt with placeholders
from langchain_google_genai import ChatGoogleGenerativeAI               # Gemini AI — the brain that generates answers

load_dotenv()   # Read .env file and load GEMINI_API_KEY into environment


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CareBot",              # Browser tab title
    page_icon="🩺",                    # Browser tab icon
    layout="wide",                     # Use full screen width
    initial_sidebar_state="expanded",  # Sidebar open by default
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
# Injects raw CSS into the page to override Streamlit's default styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
    /* ↑ Load 3 Google Fonts: DM Serif Display (headings), DM Mono (labels), Inter (body text) */

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d0f14;   /* Dark navy background for entire app */
        color: #d4cfc7;              /* Soft off-white default text color */
        font-family: 'Inter', sans-serif;   /* Apply Inter font everywhere */
    }

    [data-testid="stHeader"] { background-color: #0d0f14; }  /* Match header to background */

    h1 {
        font-family: 'DM Serif Display', serif;   /* Serif font for main headings */
        color: #e8e2d9 !important;                /* Light cream color */
        letter-spacing: 0.02em;                   /* Slight spacing between letters */
    }

    /* Style each chat message bubble */
    [data-testid="stChatMessage"] {
        background: #13161d;         /* Slightly lighter than page background */
        border: 1px solid #1e2330;   /* Subtle border */
        border-radius: 10px;         /* Rounded corners */
        padding: 18px 22px;          /* Inner spacing */
        margin-bottom: 14px;         /* Space between messages */
    }

    /* Hide default Streamlit avatars (we use our own custom ones) */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] { display: none !important; }

    /* The fixed input bar at the bottom of the screen */
    .custom-input-row {
        position: fixed;             /* Stays at bottom even when scrolling */
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 999;                /* Sits on top of all other elements */
        background: #0d0f14;
        padding: 14px 2rem 18px 2rem;
    }
    /* Style the text input box inside the fixed bar */
    .custom-input-row .stTextInput input {
        background: #13161d !important;
        color: #d4cfc7 !important;
        border: 1px solid #2a3040 !important;
        border-radius: 14px 0 0 14px !important;   /* Rounded only on left side (connects to send button) */
        padding: 12px 18px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
        box-shadow: none !important;
        outline: none !important;
        height: 52px !important;
    }
    /* Blue glow when user clicks into the input box */
    .custom-input-row .stTextInput input:focus {
        border-color: #4a9eff !important;
        box-shadow: 0 0 0 3px rgba(74,158,255,0.12) !important;
    }
    /* Greyed-out placeholder text */
    .custom-input-row .stTextInput input::placeholder {
        color: #3a4055 !important;
    }
    /* The ↑ send button styling */
    .custom-input-row .send-btn .stButton > button {
        background: #1a2a3a !important;
        color: #4a9eff !important;
        border: 1px solid #2a3040 !important;
        border-left: none !important;                  /* Merges visually with the input box */
        border-radius: 0 14px 14px 0 !important;       /* Rounded only on right side */
        height: 52px !important;
        padding: 0 18px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: background 0.18s ease !important;  /* Smooth hover animation */
    }
    .custom-input-row .send-btn .stButton > button:hover {
        background: #1e3a50 !important;   /* Slightly brighter on hover */
        color: #6ab4ff !important;
    }
    /* Mic button — default (inactive) state */
    .custom-input-row .mic-btn .stButton > button {
        background: #13161d !important;
        color: #3a5070 !important;
        border: 1px solid #2a3040 !important;
        border-left: none !important;
        border-radius: 0 !important;      /* Square — sits between input and send button */
        height: 52px !important;
        padding: 0 14px !important;
        font-size: 1rem !important;
        transition: all 0.18s ease !important;
    }
    .custom-input-row .mic-btn .stButton > button:hover {
        color: #4a9eff !important;
        background: #161b26 !important;
    }
    /* Mic button — active (recording) state, glows blue */
    .custom-input-row .mic-btn-active .stButton > button {
        color: #4a9eff !important;
        background: #0f1a28 !important;
    }
    /* Add bottom padding so chat history isn't hidden behind fixed input bar */
    .block-container {
        padding-bottom: 90px !important;
    }

    /* Sidebar dark styling */
    [data-testid="stSidebar"] {
        background-color: #0f1118;
        border-right: 1px solid #1a1e2a;
    }

    /* Custom thin scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0d0f14; }
    ::-webkit-scrollbar-thumb { background: #222536; border-radius: 4px; }

    /* General button style across the whole app */
    .stButton > button {
        background-color: #13161d;
        color: #7a9abb;
        border: 1px solid #1e2a38;
        border-radius: 6px;
        font-size: 0.78rem;
        padding: 7px 14px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        letter-spacing: 0.02em;
        transition: all 0.18s ease;
    }
    .stButton > button:hover {
        background-color: #161b26;
        border-color: #2e4a6a;
        color: #b0c8e0;
    }

    /* Metric boxes (Questions count, Messages count) */
    [data-testid="metric-container"] {
        background: #13161d;
        border: 1px solid #1e2330;
        border-radius: 8px;
        padding: 10px;
    }

    /* File upload area */
    [data-testid="stFileUploader"] {
        background: #13161d;
        border: 1px dashed #1e2a38;   /* Dashed border = classic "drop zone" look */
        border-radius: 8px;
        padding: 12px;
    }

    /* Info/warning alert boxes */
    [data-testid="stAlert"] {
        background: #111520 !important;
        border: 1px solid #1e2a38 !important;
        border-radius: 8px !important;
        color: #7a9abb !important;
    }

    /* Top and bottom padding for main content area */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,   # Allow raw HTML/CSS to be injected
)

# ── AVATAR HTML ───────────────────────────────────────────────────────────────
# These are small HTML snippets shown above each chat message
# They replace Streamlit's default avatar icons

# Avatar shown above user messages — grey circle with 👤 icon
USER_AVATAR_HTML = """
<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
    <div style="width:28px;height:28px;border-radius:50%;background:#161b26;
        border:1px solid #2a3040;display:flex;align-items:center;justify-content:center;
        font-size:12px;flex-shrink:0;box-shadow:0 0 8px rgba(100,120,200,0.18);">👤</div>
    <span style="font-size:0.7rem;color:#4a5570;letter-spacing:0.1em;text-transform:uppercase;font-family:'Inter',sans-serif;font-weight:600;">You</span>
</div>
"""

# Avatar shown above CareBot messages — blue circle with + icon
BOT_AVATAR_HTML = """
<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
    <div style="width:28px;height:28px;border-radius:50%;background:#0f1a28;
        border:1px solid #2a4060;display:flex;align-items:center;justify-content:center;
        font-size:12px;flex-shrink:0;box-shadow:0 0 10px rgba(40,100,180,0.25);">+</div>
    <span style="font-size:0.7rem;color:#2e5070;letter-spacing:0.1em;text-transform:uppercase;font-family:'Inter',sans-serif;font-weight:600;">CareBot</span>
</div>
"""

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR = "chroma_db"       # Folder where the pre-built vector DB lives on disk
CHROMA_COLLECTION_NAME = "chatbot"     # Name of the collection inside Chroma
DATA_PATH = "data"                     # Folder where base medical PDFs/TXTs are stored

# The instruction template sent to Gemini with every question
# {context} = retrieved document chunks | {question} = user's question
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know — don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# The 4 quick-tap suggestion cards shown on empty chat screen
# Each tuple is: ("Card Label", "Full question text")
QUICK_SUGGESTIONS = [
    ("Symptoms", "What are common symptoms of diabetes?"),
    ("Blood Pressure", "How to manage high blood pressure?"),
    ("Fatigue", "What causes persistent fatigue?"),
    ("Mental Health", "What are early signs of anxiety?"),
]


# ── DOCUMENT LOADING ──────────────────────────────────────────────────────────

def load_documents(data_dir: str = DATA_PATH):
    data_path = Path(data_dir)
    if not data_path.exists():                              # Stop if the data folder doesn't exist
        raise RuntimeError(f"Data directory not found at '{data_dir}'.")
    documents = []
    for path in sorted(data_path.rglob("*.txt")):          # Find all .txt files recursively
        documents.extend(TextLoader(str(path)).load())     # Load each .txt into LangChain Document objects
    for path in sorted(data_path.rglob("*.pdf")):          # Find all .pdf files recursively
        documents.extend(PyPDFLoader(str(path)).load())    # Load each .pdf into LangChain Document objects
    if not documents:
        raise RuntimeError(f"No documents found in '{data_dir}'.")
    return documents   # Returns a list of LangChain Document objects


# ── VECTOR STORE BUILDERS ─────────────────────────────────────────────────────

@st.cache_resource    # Streamlit runs this only ONCE and reuses the result across reruns
def build_vectorstore(source_docs=None):
    if source_docs is None:
        source_docs = load_documents()    # Load from disk if no docs passed in
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # ↑ Load the model that converts text → numeric vectors
    return Chroma.from_documents(
        source_docs, embedding_model,          # Embed all documents
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,  # Save to disk so we don't rebuild every time
    )


@st.cache_resource     # Also cached — loads existing DB from disk instead of rebuilding
def get_vectorstore():
    if not os.path.exists(CHROMA_PERSIST_DIR):   # If DB doesn't exist yet, build it fresh
        return build_vectorstore()
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,      # Need same embedding model used to build it
        persist_directory=CHROMA_PERSIST_DIR,    # Point to saved DB on disk
    )


# ── PROMPT & QA CHAIN ─────────────────────────────────────────────────────────

def set_custom_prompt(template):
    # Wraps the template string into a LangChain PromptTemplate object
    return PromptTemplate(template=template, input_variables=["context", "question"])


def build_qa_chain(vectorstore):
    gemini_api_key = os.environ.get("GEMINI_API_KEY")   # Fetch API key from environment
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.0,        # temperature=0 = factual, no creativity
        max_tokens=None, timeout=None, max_retries=2, api_key=gemini_api_key,
        # max_tokens=None: no answer length limit | max_retries=2: retry on failure
    )
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",                                         # "stuff" = paste all chunks together into one prompt
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),          # Fetch top 3 most relevant chunks from DB
        return_source_documents=True,                                         # Also return which documents were used
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},  # Pass our custom instructions to the AI
    )


def build_qa_chain_with_custom_store(vectorstore):   # Wrapper — just calls build_qa_chain with a different store
    return build_qa_chain(vectorstore)


# ── SOURCE FORMATTER ──────────────────────────────────────────────────────────

def format_source_documents(source_documents):     # Turns retrieved doc metadata into a readable source list
    if not source_documents:
        return ""    # Nothing to show if no sources returned
    lines = []
    for doc in source_documents:
        source = doc.metadata.get("source", "unknown")   # Get filename; fallback to "unknown" if missing
        page = doc.metadata.get("page", "")              # Get page number; empty string if missing
        lines.append(f"- {source} (page {page})" if page else f"- {source}")
        # ↑ Ternary: include page number only if it exists
    return "\n\n**Sources:**\n" + "\n".join(lines)   # Format as a markdown bullet list


# ── SPEECH TO TEXT ────────────────────────────────────────────────────────────

def audio_to_text(audio_bytes):
    if not HAS_SPEECH_RECOGNITION:          # Safety check — don't crash if library missing
        st.error("Speech recognition not available.")
        return None
    try:
        recognizer = sr.Recognizer()                  # Create the speech recognition engine
        audio_io = io.BytesIO(audio_bytes)            # Convert raw audio bytes → file-like object
        with sr.AudioFile(audio_io) as source:        # Open the audio as a file
            audio_data = recognizer.record(source)    # Read the full audio clip
        return recognizer.recognize_google(audio_data)  # Send to Google → get back text string
    except sr.UnknownValueError:
        st.warning("Could not understand audio. Please try again.")   # Audio was unclear
        return None
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")   # Any other error
        return None


# ── MESSAGE RENDERER ──────────────────────────────────────────────────────────

def render_message(role: str, content: str):
    avatar_html = USER_AVATAR_HTML if role == "User" else BOT_AVATAR_HTML   # Pick correct avatar
    with st.chat_message(role):            # Opens a styled Streamlit chat bubble
        st.markdown(avatar_html, unsafe_allow_html=True)   # Render the custom avatar HTML
        st.markdown(content)               # Render the message text (supports markdown)


# ── ANSWER DISPATCHER ─────────────────────────────────────────────────────────

def get_answer(prompt, use_uploaded_files, temp_vectorstore):
    if use_uploaded_files and temp_vectorstore:
        # User checked "Answer from uploaded docs only" — search ONLY uploaded files
        qa_chain = build_qa_chain_with_custom_store(temp_vectorstore)
        response = qa_chain.invoke({"query": prompt})
        result = response.get("result") or response.get("output_text") or ""
        sources = format_source_documents(response.get("source_documents", []))
        return result + sources

    elif temp_vectorstore:
        # Files uploaded but checkbox is OFF — search BOTH uploaded docs AND base DB
        qa_temp = build_qa_chain_with_custom_store(temp_vectorstore)
        r_temp = qa_temp.invoke({"query": prompt})          # Answer from uploaded files
        qa_base = build_qa_chain(get_vectorstore())
        r_base = qa_base.invoke({"query": prompt})          # Answer from base knowledge DB
        result = (
            f"**From Uploaded Documents:**\n{r_temp.get('result','')}\n\n"
            f"**From Base Knowledge:**\n{r_base.get('result','')}"
        )
        all_docs = r_temp.get("source_documents", []) + r_base.get("source_documents", [])  # Combine sources from both
        return result + format_source_documents(all_docs)

    else:
        # No files uploaded — search only the base Chroma DB
        qa_chain = build_qa_chain(get_vectorstore())
        response = qa_chain.invoke({"query": prompt})
        result = response.get("result") or response.get("output_text") or ""
        sources = format_source_documents(response.get("source_documents", []))
        return result + sources


# ── INPUT SUBMIT HANDLER ──────────────────────────────────────────────────────

def _submit_input():
    # Called when user presses Enter or clicks ↑ button
    val = st.session_state.get("chat_text_input", "").strip()   # Read current input box value
    if val:
        st.session_state.pending_prompt = val   # Store it so main() can process it
    st.session_state["chat_text_input"] = ""    # Clear the input box after submission


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION — Builds and runs the entire UI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Session state init ────────────────────────────────────────────────────
    # Session state = Streamlit's memory that persists across reruns
    # Initialize each key with a default value if it doesn't exist yet
    for key, default in [
        ("messages", []),           # List of all chat messages {role, content}
        ("suggested_prompt", None), # Holds a prompt selected from suggestion cards
        ("total_questions", 0),     # Counter for how many questions asked
        ("show_voice", False),      # Whether the mic recording UI is visible
        ("pending_prompt", None),   # Prompt waiting to be processed this rerun
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:   # Everything inside this block appears in the left sidebar
        st.markdown(
            """
            <div style="padding: 20px 4px 12px 4px;">
                <div style="font-family:'DM Serif Display',serif;font-size:1.35rem;color:#c8c2b8;
                    letter-spacing:0.04em;">CareBot</div>
                <div style="height:1px;background:#1a1e2a;margin-top:10px;"></div>
            </div>
            """,
            unsafe_allow_html=True,   # Sidebar title + divider line
        )

        if st.button("New Chat", use_container_width=True, key="new_chat_btn"):
            # Reset everything to start a fresh conversation
            st.session_state.messages = []
            st.session_state.total_questions = 0
            st.session_state.pending_prompt = None
            st.rerun()   # Force Streamlit to re-render the page immediately

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)   # Spacer
        st.markdown(
            """
            <div style="font-size:0.7rem;color:#3a4055;letter-spacing:0.1em;
                text-transform:uppercase;margin-top:16px;margin-bottom:8px;
                font-family:'Inter',sans-serif;font-weight:600;">Upload Report</div>
            """,
            unsafe_allow_html=True,   # "UPLOAD REPORT" section label
        )
        st.caption("PDF or TXT files — your documents stay private")   # Small hint text below label

        # File upload widget — accepts multiple PDF or TXT files
        uploaded_files = st.file_uploader(
            "Choose files", type=["pdf", "txt"],
            accept_multiple_files=True, label_visibility="collapsed",   # Hide default label
        )

        use_uploaded_files = False    # Default: don't restrict to uploaded docs only
        temp_vectorstore = None       # Default: no uploaded docs vector store

        if uploaded_files:   # Only runs if user actually uploaded something
            st.success(f"{len(uploaded_files)} file(s) ready")
            with st.spinner("Processing..."):   # Show loading spinner while processing
                temp_docs = []
                temp_dir = tempfile.mkdtemp()   # Create a temporary folder on disk
                for f in uploaded_files:
                    fpath = os.path.join(temp_dir, f.name)   # Build full path for this file
                    with open(fpath, "wb") as out:
                        out.write(f.getbuffer())    # Save the uploaded file bytes to disk
                    try:
                        if f.name.endswith(".pdf"):
                            temp_docs.extend(PyPDFLoader(fpath).load())    # Load PDF pages
                        elif f.name.endswith(".txt"):
                            temp_docs.extend(TextLoader(fpath).load())     # Load TXT content
                    except Exception as e:
                        st.warning(f"Could not read {f.name}: {e}")        # Skip bad files gracefully

                if temp_docs:
                    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    temp_vectorstore = Chroma.from_documents(temp_docs, emb, collection_name="temp_uploads")
                    # ↑ Build a temporary in-memory vector store from uploaded docs (no disk save)
                    use_uploaded_files = st.checkbox(
                        "Answer from uploaded docs only",
                        value=False,   # Unchecked by default → searches both sources
                    )

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)   # Spacer
        st.markdown(
            "<div style='height:1px;background:#1a1e2a;margin-bottom:12px;'></div>",
            unsafe_allow_html=True,   # Horizontal divider line
        )

        # Show two stat counters side by side
        col_a, col_b = st.columns(2)
        col_a.metric("Questions", st.session_state.total_questions)   # How many questions asked
        col_b.metric("Messages", len(st.session_state.messages))      # Total messages in chat

        if st.session_state.messages:   # Only show export button if there's something to export
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            chat_data = {
                "exported_at": datetime.now().isoformat(),                 # Timestamp of export
                "total_questions": st.session_state.total_questions,
                "messages": st.session_state.messages,                     # Full chat history
            }
            st.download_button(
                label="Export chat",
                data=json.dumps(chat_data, indent=2),   # Convert dict to formatted JSON string
                file_name=f"carebot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",  # Timestamped filename
                mime="application/json",
                use_container_width=True,
                key="download_btn",
            )

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)   # Spacer
        st.markdown(
            """
            <div style="height:1px;background:#1a1e2a;margin-bottom:16px;"></div>
            <div style="display:flex;flex-direction:column;gap:10px;margin-bottom:16px;">
                <!-- Green checkmark: Evidence-based information -->
                <div style="display:flex;align-items:center;gap:9px;">
                    <div style="width:15px;height:15px;border:1.5px solid #2a4a2a;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                        <div style="width:6px;height:4px;border-left:1.5px solid #3a7a3a;
                            border-bottom:1.5px solid #3a7a3a;transform:rotate(-45deg);margin-top:-1px;"></div>
                    </div>
                    <span style="font-size:0.74rem;color:#3a5040;font-family:'Inter',sans-serif;">Evidence-based information</span>
                </div>
                <!-- Green checkmark: Private document analysis -->
                <div style="display:flex;align-items:center;gap:9px;">
                    <div style="width:15px;height:15px;border:1.5px solid #2a4a2a;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                        <div style="width:6px;height:4px;border-left:1.5px solid #3a7a3a;
                            border-bottom:1.5px solid #3a7a3a;transform:rotate(-45deg);margin-top:-1px;"></div>
                    </div>
                    <span style="font-size:0.74rem;color:#3a5040;font-family:'Inter',sans-serif;">Private document analysis</span>
                </div>
                <!-- Blue dot: disclaimer — not professional care -->
                <div style="display:flex;align-items:center;gap:9px;">
                    <div style="width:15px;height:15px;border:1.5px solid #2a3a4a;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                        <div style="width:5px;height:5px;border:1.5px solid #3a5a7a;border-radius:50%;"></div>
                    </div>
                    <span style="font-size:0.74rem;color:#2e3d4a;font-family:'Inter',sans-serif;">Not a replacement for professional care</span>
                </div>
            </div>
            <!-- Fine print medical disclaimer -->
            <div style="font-size:0.65rem;color:#222630;line-height:1.6;font-family:'Inter',sans-serif;padding:0 2px;">
                For informational use only. Not a substitute for professional medical advice.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── HEADER — always visible, regardless of chat history ──────────────────
    # Renders the robot SVG icon + "ASK CAREBOT" title + subtitle
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:22px;margin-bottom:4px;margin-top:8px;">
            <div style="flex-shrink:0;">
                <!-- Custom robot SVG illustration — built with basic shapes -->
                <svg width="72" height="72" viewBox="0 0 72 72" fill="none">
                    <line x1="36" y1="6" x2="36" y2="16" stroke="#3a6090" stroke-width="2.5" stroke-linecap="round"/>
                    <circle cx="36" cy="4.5" r="3" fill="#4a80b0" opacity="0.9"/>
                    <rect x="16" y="16" width="40" height="30" rx="9" fill="#13161d" stroke="#2a4060" stroke-width="1.8"/>
                    <rect x="22" y="24" width="10" height="7" rx="3" fill="#1a3a5c"/>
                    <rect x="40" y="24" width="10" height="7" rx="3" fill="#1a3a5c"/>
                    <circle cx="27" cy="27.5" r="2.5" fill="#4a9eff" opacity="0.95"/>
                    <circle cx="45" cy="27.5" r="2.5" fill="#4a9eff" opacity="0.95"/>
                    <rect x="25" y="36" width="22" height="5" rx="2.5" fill="#1a3a5c"/>
                    <rect x="27" y="37.5" width="4" height="2" rx="1" fill="#4a9eff" opacity="0.8"/>
                    <rect x="33" y="37.5" width="4" height="2" rx="1" fill="#4a9eff" opacity="0.8"/>
                    <rect x="39" y="37.5" width="4" height="2" rx="1" fill="#4a9eff" opacity="0.8"/>
                    <rect x="8" y="22" width="8" height="14" rx="4" fill="#13161d" stroke="#2a4060" stroke-width="1.5"/>
                    <rect x="56" y="22" width="8" height="14" rx="4" fill="#13161d" stroke="#2a4060" stroke-width="1.5"/>
                    <rect x="20" y="48" width="32" height="18" rx="6" fill="#13161d" stroke="#2a4060" stroke-width="1.8"/>
                    <rect x="33" y="52" width="6" height="10" rx="1.5" fill="#2a5080" opacity="0.9"/>
                    <rect x="30" y="55" width="12" height="4" rx="1.5" fill="#2a5080" opacity="0.9"/>
                    <rect x="22" y="66" width="10" height="5" rx="2.5" fill="#13161d" stroke="#2a4060" stroke-width="1.5"/>
                    <rect x="40" y="66" width="10" height="5" rx="2.5" fill="#13161d" stroke="#2a4060" stroke-width="1.5"/>
                </svg>
            </div>
            <div>
                <!-- Main title -->
                <div style="font-family:'DM Serif Display',serif;font-size:2.6rem;line-height:1.1;
                    color:#e8e2d9;letter-spacing:0.04em;">ASK CAREBOT</div>
                <!-- Subtitle in monospace font -->
                <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#3a6090;
                    letter-spacing:0.18em;text-transform:uppercase;margin-top:5px;">
                    YOUR PERSONAL AI HEALTH ASSISTANT</div>
            </div>
        </div>
        <!-- Gradient underline below header -->
        <div style="height:1px;background:linear-gradient(90deg,#2a4060 0%,transparent 80%);margin-bottom:18px;"></div>
        """,
        unsafe_allow_html=True,
    )

    # Show info banner based on which knowledge source is active
    if use_uploaded_files and temp_vectorstore:
        st.info("Answering from your uploaded documents only.")
    elif temp_vectorstore:
        st.info("Searching both uploaded documents and base knowledge.")

    has_messages = bool(st.session_state.messages)   # True if chat history exists

    # ── Quick Suggestion Cards — only on empty chat ───────────────────────────
    if not has_messages:   # Only show suggestion cards when no messages yet
        st.markdown(
            "<div style='font-size:0.7rem;color:#4a9eff;text-shadow:0 0 8px rgba(74,158,255,0.5);letter-spacing:0.1em;text-transform:uppercase;"
            "margin-bottom:14px;font-family:\"Inter\",sans-serif;font-weight:600;'>Common Questions</div>",
            unsafe_allow_html=True,   # "COMMON QUESTIONS" section label
        )

        cols = st.columns(4)   # Create 4 equal columns for the 4 suggestion cards
        card_css = """
            background:#111520;
            border:1px solid #1a2030;
            border-radius:16px;
            padding:20px 16px;
            cursor:pointer;
            transition:border-color 0.18s ease, background 0.18s ease;
            min-height:90px;
            display:flex;
            flex-direction:column;
            justify-content:flex-end;
        """
        # ↑ CSS for each card: dark background, rounded, content aligned to bottom

        for i, (label, question) in enumerate(QUICK_SUGGESTIONS):   # Loop through all 4 suggestions with index
            with cols[i]:   # Place each card in its own column
                st.markdown(
                    f"""
                    <div style="{card_css}">
                        <div style="font-size:0.68rem;color:#2e3a4a;text-transform:uppercase;
                            letter-spacing:0.1em;font-family:'Inter',sans-serif;font-weight:600;
                            margin-bottom:6px;">{label}</div>
                        <!-- ↑ Small uppercase label e.g. "SYMPTOMS" -->
                        <div style="font-size:0.82rem;color:#7a8a9a;font-family:'Inter',sans-serif;
                            line-height:1.4;">{question}</div>
                        <!-- ↑ Full question text displayed below the label -->
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Ask", key=f"sug_{i}", use_container_width=True):
                    # When "Ask" clicked: store the question as pending and rerun
                    st.session_state.pending_prompt = question
                    st.rerun()

        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)   # Bottom spacer

    # ── Conversation history ──────────────────────────────────────────────────
    for message in st.session_state.messages:   # Loop through all stored messages and render each
        render_message(message["role"], message["content"])

    # Invisible anchor element at the bottom of the chat
    st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <script>
            const el = document.getElementById('chat-bottom');
            if (el) el.scrollIntoView({ behavior: 'smooth' });
        </script>
        """,
        unsafe_allow_html=True,   # Auto-scroll to latest message after each rerun
    )

    # ── Fixed input bar ───────────────────────────────────────────────────────
    st.markdown('<div class="custom-input-row">', unsafe_allow_html=True)   # Open fixed bar wrapper

    # Layout: if mic available → 3 columns (text | mic | send), else → 2 columns (text | send)
    if HAS_SPEECH_RECOGNITION:
        col_text, col_mic, col_send = st.columns([11, 1, 1])   # Wide text, narrow mic, narrow send
    else:
        col_text, col_send = st.columns([12, 1])
        col_mic = None

    with col_text:
        st.text_input(
            "input",
            label_visibility="collapsed",           # Hide the label text
            placeholder="Ask a health question...", # Grey hint text inside box
            key="chat_text_input",                  # Session state key to read value from
            on_change=_submit_input,                # Called automatically when user presses Enter
        )

    if col_mic:
        mic_class = "mic-btn-active" if st.session_state.show_voice else "mic-btn"  # Toggle active style
        st.markdown(f'<div class="{mic_class}">', unsafe_allow_html=True)
        with col_mic:
            if st.button("🎙", key="mic_toggle"):
                st.session_state.show_voice = not st.session_state.show_voice   # Toggle mic UI on/off
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="send-btn">', unsafe_allow_html=True)
    with col_send:
        if st.button("↑", key="send_btn", on_click=_submit_input):   # ↑ button also calls submit
            pass
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)   # Close fixed bar wrapper

    # ── Voice input (original speech_recognition) ─────────────────────────────
    if HAS_SPEECH_RECOGNITION and st.session_state.show_voice:   # Only show if mic is toggled on
        audio_data = st.audio_input("Speak now", label_visibility="visible")   # Show mic recorder widget
        if audio_data:
            with st.spinner("Converting speech to text..."):
                voice_prompt = audio_to_text(audio_data.getvalue())   # Convert audio bytes → text
                if voice_prompt:
                    st.success(f"Heard: **{voice_prompt}**")               # Show what was heard
                    st.session_state.pending_prompt = voice_prompt         # Queue it as next prompt
                    st.session_state.show_voice = False                    # Hide mic UI after capture

    # ── Resolve the final prompt ──────────────────────────────────────────────
    prompt = None
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt       # Grab the waiting prompt
        st.session_state.pending_prompt = None         # Clear it so it doesn't repeat next rerun

    # ── Process the prompt ────────────────────────────────────────────────────
    if prompt:
        st.session_state.messages.append({"role": "User", "content": prompt})   # Save user message
        st.session_state.total_questions += 1                                    # Increment counter

        render_message("User", prompt)   # Display user message immediately

        with st.spinner("CareBot is thinking..."):   # Show spinner while waiting for AI
            try:
                answer = get_answer(prompt, use_uploaded_files, temp_vectorstore)   # Get AI answer
            except Exception as e:
                answer = f"Something went wrong: {str(e)}"   # Show error gracefully

        st.session_state.messages.append({"role": "assistant", "content": answer})   # Save bot reply
        render_message("assistant", answer)   # Display bot reply
        st.rerun()   # Rerun page to update metrics and scroll to bottom


if __name__ == "__main__":
    main()   # Entry point — run the app