import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import json
import tempfile
from datetime import datetime
import io

try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False

from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(
    page_title="CareBot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────
# Palette built for a clinical, trustworthy tone with real contrast (WCAG AA+):
#   Canvas   #0B1220  — base background
#   Surface  #121B2E  — cards, chat bubbles
#   Surface+ #182338  — elevated / hover surfaces
#   Line     #263248  — borders / hairlines
#   Ink      #EAF0F8  — primary text (on Canvas/Surface: ~13:1 contrast)
#   Ink-dim  #AAB8CC  — secondary text (~6.5:1 contrast)
#   Ink-mute #7C8AA0  — tertiary / meta text (~4.6:1 contrast, still AA)
#   Teal     #35D6C0  — clinical accent (bot, links, focus)
#   Amber    #F0B429  — warm accent (user, warnings)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --canvas: #0B1220;
        --surface: #121B2E;
        --surface-hi: #182338;
        --line: #263248;
        --ink: #EAF0F8;
        --ink-dim: #AAB8CC;
        --ink-mute: #7C8AA0;
        --teal: #35D6C0;
        --teal-dim: #1B4A44;
        --amber: #F0B429;
        --amber-dim: #4A3A15;

        /* Fallback sidebar width, used until JS measures the real value.
           This matches Streamlit's default expanded sidebar width, so the
           very first paint already reserves the right amount of space and
           the input bar never has to "snap" into place. */
        --sidebar-w: 21rem;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: var(--canvas) !important;
        color: var(--ink) !important;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] { background-color: var(--canvas) !important; }

    /* Make every default text element inherit a legible color instead of
       falling back to Streamlit's own (too-dark-on-dark) defaults. */
    p, span, li, label, div[data-testid="stMarkdownContainer"] {
        color: var(--ink) !important;
    }

    h1, h2, h3 {
        font-family: 'Source Serif 4', serif;
        color: var(--ink) !important;
        letter-spacing: 0.01em;
    }

    /* ── Chat bubbles ──────────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    [data-testid="stChatMessage"] p {
        color: var(--ink) !important;
        font-size: 0.98rem;
        line-height: 1.65;
    }
    [data-testid="stChatMessage"] strong { color: var(--ink) !important; }

    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] { display: none !important; }

    /* ── Fixed input bar ───────────────────────────────────────────────── *
     * `left` tracks the `--sidebar-w` CSS variable instead of a value set
     * directly on this element. That distinction matters: Streamlit
     * recreates this container's DOM node on every rerun, so any inline
     * style JS applied straight to it is lost the instant it reruns, which
     * is what let the bar render at `left: 0` (spilling under/over the
     * open sidebar) right after a rerun, before the old sync script caught
     * up. A CSS variable set on <html> below survives reruns because the
     * root element itself is never recreated — every new instance of this
     * bar inherits the correct offset immediately, with no flash and no
     * dependence on re-querying this specific node in time.
     */
    .st-key-custom_input_row {
        position: fixed;
        bottom: 0;
        left: var(--sidebar-w);
        right: 0;
        width: auto;
        box-sizing: border-box;
        z-index: 999;
        background: linear-gradient(180deg, rgba(11,18,32,0) 0%, var(--canvas) 35%);
        padding: 22px 2rem 20px 2rem;
        transition: left 0.22s ease;
    }
    .st-key-custom_input_row .stTextInput input {
        background: var(--surface) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px 0 0 12px !important;
        padding: 12px 18px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        box-shadow: none !important;
        outline: none !important;
        height: 52px !important;
    }
    .st-key-custom_input_row .stTextInput input:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(53,214,192,0.18) !important;
    }
    .st-key-custom_input_row .stTextInput input::placeholder {
        color: var(--ink-mute) !important;
    }
    .st-key-custom_input_row .send-btn .stButton > button {
        background: var(--teal-dim) !important;
        color: var(--teal) !important;
        border: 1px solid var(--line) !important;
        border-left: none !important;
        border-radius: 0 12px 12px 0 !important;
        height: 52px !important;
        padding: 0 20px !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        transition: background 0.18s ease !important;
    }
    .st-key-custom_input_row .send-btn .stButton > button:hover {
        background: #235e56 !important;
        color: #7ff0e0 !important;
    }
    .st-key-custom_input_row .mic-btn .stButton > button,
    .st-key-custom_input_row .mic-btn-active .stButton > button {
        background: var(--surface) !important;
        color: var(--ink-dim) !important;
        border: 1px solid var(--line) !important;
        border-left: none !important;
        border-radius: 0 !important;
        height: 52px !important;
        padding: 0 16px !important;
        font-size: 1.05rem !important;
        transition: all 0.18s ease !important;
    }
    .st-key-custom_input_row .mic-btn-active .stButton > button {
        color: var(--teal) !important;
        background: var(--teal-dim) !important;
        border-color: var(--teal) !important;
    }

    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 100px !important;
    }

    /* ── Sidebar ───────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #0E1526 !important;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * {
        color: var(--ink-dim) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
        color: var(--ink-mute) !important;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--canvas); }
    ::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }

    /* ── Buttons (general) ────────────────────────────────────────────── */
    .stButton > button {
        background-color: var(--surface);
        color: var(--ink) !important;
        border: 1px solid var(--line);
        border-radius: 8px;
        font-size: 0.85rem;
        padding: 9px 16px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        transition: all 0.18s ease;
    }
    .stButton > button:hover {
        background-color: var(--surface-hi);
        border-color: var(--teal);
        color: var(--teal) !important;
    }
    .stButton > button p { color: inherit !important; }

    /* ── Metrics ───────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 12px;
    }
    [data-testid="stMetricValue"] { color: var(--ink) !important; font-size: 1.6rem; }
    [data-testid="stMetricLabel"] { color: var(--ink-mute) !important; }

    /* ── File uploader ─────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--surface);
        border: 1.5px dashed var(--line);
        border-radius: 10px;
        padding: 14px;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--ink-dim) !important;
    }
    [data-testid="stBaseButton-secondary"] {
        background: var(--surface-hi) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
    }

    /* ── Alerts (info/success/warning) ────────────────────────────────── */
    [data-testid="stAlertContentInfo"] { color: var(--ink) !important; }
    [data-testid="stAlertContentSuccess"] { color: var(--ink) !important; }
    [data-testid="stAlert"] {
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
    }
    [data-testid="stAlert"] p { color: var(--ink) !important; }

    /* ── Checkbox label ────────────────────────────────────────────────── */
    [data-testid="stCheckbox"] label p { color: var(--ink-dim) !important; }

    /* ── Header eyebrow / suggestion cards ────────────────────────────── */
    .section-label {
        font-size: 0.75rem;
        color: var(--teal);
        letter-spacing: 0.06em;
        margin-bottom: 14px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }

    /* Force the suggestion-card row to stretch every column to the same
       height (the tallest one), then let each card grow to fill its
       column's leftover space. This replaces the old fixed `min-height`,
       which broke as soon as one question wrapped to a second line. */
    div[data-testid="stHorizontalBlock"]:has(.suggestion-card) {
        align-items: stretch;
    }
    div[data-testid="stHorizontalBlock"]:has(.suggestion-card) > div[data-testid="column"] {
        display: flex;
    }
    div[data-testid="stHorizontalBlock"]:has(.suggestion-card) > div[data-testid="column"] > div {
        display: flex;
        flex-direction: column;
        width: 100%;
    }
    div[data-testid="stHorizontalBlock"]:has(.suggestion-card) [data-testid="stVerticalBlock"] {
        height: 100%;
    }
    .suggestion-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px 16px;
        flex: 1 1 auto;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        margin-bottom: 10px;
        transition: border-color 0.18s ease, background 0.18s ease;
    }
    .suggestion-card .cat {
        font-size: 0.7rem;
        color: var(--teal);
        font-weight: 700;
        margin-bottom: 6px;
    }
    .suggestion-card .q {
        font-size: 0.86rem;
        color: var(--ink-dim);
        line-height: 1.4;
    }

    /* ── Conversation anchor ──────────────────────────────────────────── *
     * Real chat apps (Claude, ChatGPT) keep a short conversation pinned
     * just above the composer instead of floating at the top of the page.
     * `justify-content: flex-end` inside a viewport-relative min-height
     * box does exactly that: messages stack from the bottom up, and once
     * they exceed the box's height, normal top-down flow / scrolling
     * takes over automatically.
     */
    .st-key-chat_messages_anchor {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        min-height: calc(100vh - 180px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar-aware positioning for the fixed input bar ───────────────────────
# Instead of writing the measured width directly onto the input bar element
# (`.st-key-custom_input_row`), this sets a CSS variable, `--sidebar-w`, on
# <html>. That's the key fix: Streamlit tears down and recreates the input
# bar's DOM node on every rerun, so inline styles applied straight to it
# only last until the next rerun — there's a window where the freshly
# recreated node has no inline style yet and falls back to `left: 0`,
# which is exactly what let it render underneath/over an open sidebar.
# <html> is never recreated, so the variable — and therefore the correct
# offset — survives every rerun automatically, with no re-sync needed.
st.markdown(
    """
    <script>
    (function () {
        if (window.__carebotInputSyncInit) { return; }
        window.__carebotInputSyncInit = true;

        const root = document.documentElement;

        function syncInputBar() {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) {
                root.style.setProperty('--sidebar-w', '0px');
                return;
            }
            const expanded = sidebar.getAttribute('aria-expanded') !== 'false';
            if (!expanded) {
                root.style.setProperty('--sidebar-w', '0px');
                return;
            }
            const width = sidebar.getBoundingClientRect().width;
            // Guard against transient 0-width reads (e.g. mid-layout, before
            // fonts/content settle) so we never overwrite a good value with
            // a bad one and cause the bar to jump under the sidebar.
            if (width > 0) {
                root.style.setProperty('--sidebar-w', width + 'px');
            }
        }

        syncInputBar();
        window.addEventListener('resize', syncInputBar);

        const attachObservers = () => {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar && !sidebar.__carebotObserved) {
                sidebar.__carebotObserved = true;
                new ResizeObserver(syncInputBar).observe(sidebar);
                new MutationObserver(syncInputBar).observe(sidebar, {
                    attributes: true,
                    attributeFilter: ['aria-expanded', 'style'],
                });
            }
        };
        attachObservers();

        // Streamlit swaps DOM nodes on rerun; keep re-checking so the sync
        // survives reruns instead of silently going stale.
        new MutationObserver(() => {
            attachObservers();
            syncInputBar();
        }).observe(document.body, { childList: true, subtree: true });
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

USER_AVATAR_HTML = """
<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
    <div style="width:26px;height:26px;border-radius:50%;background:#4A3A15;
        border:1px solid #6b4f1c;display:flex;align-items:center;justify-content:center;
        font-size:13px;flex-shrink:0;color:#F0B429;font-weight:700;">Y</div>
    <span style="font-size:0.72rem;color:#F0B429;letter-spacing:0.04em;font-family:'Inter',sans-serif;font-weight:600;">You</span>
</div>
"""

BOT_AVATAR_HTML = """
<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
    <div style="width:26px;height:26px;border-radius:50%;background:#1B4A44;
        border:1px solid #2b6a61;display:flex;align-items:center;justify-content:center;
        font-size:13px;flex-shrink:0;color:#35D6C0;font-weight:700;">+</div>
    <span style="font-size:0.72rem;color:#35D6C0;letter-spacing:0.04em;font-family:'Inter',sans-serif;font-weight:600;">CareBot</span>
</div>
"""

CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "chatbot"
DATA_PATH = "data"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know — don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

QUICK_SUGGESTIONS = [
    ("Symptoms", "What are common symptoms of diabetes?"),
    ("Blood Pressure", "How to manage high blood pressure?"),
    ("Fatigue", "What causes persistent fatigue?"),
    ("Mental Health", "What are early signs of anxiety?"),
]


def load_documents(data_dir: str = DATA_PATH):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise RuntimeError(f"Data directory not found at '{data_dir}'.")
    documents = []
    for path in sorted(data_path.rglob("*.txt")):
        documents.extend(TextLoader(str(path)).load())
    for path in sorted(data_path.rglob("*.pdf")):
        documents.extend(PyPDFLoader(str(path)).load())
    if not documents:
        raise RuntimeError(f"No documents found in '{data_dir}'.")
    return documents


@st.cache_resource
def build_vectorstore(source_docs=None):
    if source_docs is None:
        source_docs = load_documents()
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(
        source_docs, embedding_model,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )


@st.cache_resource
def get_vectorstore():
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return build_vectorstore()
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])


def build_qa_chain(vectorstore):
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.0,
        max_tokens=None, timeout=None, max_retries=2, api_key=gemini_api_key,
    )
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    )


def build_qa_chain_with_custom_store(vectorstore):
    return build_qa_chain(vectorstore)


def format_source_documents(source_documents):
    if not source_documents:
        return ""
    lines = []
    for doc in source_documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        lines.append(f"- {source} (page {page})" if page else f"- {source}")
    return "\n\n**Sources:**\n" + "\n".join(lines)


def audio_to_text(audio_bytes):
    if not HAS_SPEECH_RECOGNITION:
        st.error("Speech recognition not available.")
        return None
    try:
        recognizer = sr.Recognizer()
        audio_io = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_io) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        st.warning("Could not understand audio. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None


def render_message(role: str, content: str):
    avatar_html = USER_AVATAR_HTML if role == "User" else BOT_AVATAR_HTML
    with st.chat_message(role):
        st.markdown(avatar_html, unsafe_allow_html=True)
        st.markdown(content)


def get_answer(prompt, use_uploaded_files, temp_vectorstore):
    if use_uploaded_files and temp_vectorstore:
        qa_chain = build_qa_chain_with_custom_store(temp_vectorstore)
        response = qa_chain.invoke({"query": prompt})
        result = response.get("result") or response.get("output_text") or ""
        sources = format_source_documents(response.get("source_documents", []))
        return result + sources
    elif temp_vectorstore:
        qa_temp = build_qa_chain_with_custom_store(temp_vectorstore)
        r_temp = qa_temp.invoke({"query": prompt})
        qa_base = build_qa_chain(get_vectorstore())
        r_base = qa_base.invoke({"query": prompt})
        result = (
            f"**From Uploaded Documents:**\n{r_temp.get('result','')}\n\n"
            f"**From Base Knowledge:**\n{r_base.get('result','')}"
        )
        all_docs = r_temp.get("source_documents", []) + r_base.get("source_documents", [])
        return result + format_source_documents(all_docs)
    else:
        qa_chain = build_qa_chain(get_vectorstore())
        response = qa_chain.invoke({"query": prompt})
        result = response.get("result") or response.get("output_text") or ""
        sources = format_source_documents(response.get("source_documents", []))
        return result + sources


# ── Callback: captures typed text before the widget resets ───────────────────
def _submit_input():
    """Called by on_change when Enter is pressed or send is clicked."""
    val = st.session_state.get("chat_text_input", "").strip()
    if val:
        st.session_state.pending_prompt = val
    st.session_state["chat_text_input"] = ""


def main():
    for key, default in [
        ("messages", []),
        ("suggested_prompt", None),
        ("total_questions", 0),
        ("show_voice", False),
        ("pending_prompt", None),
        ("awaiting_answer", False),
        ("awaiting_prompt", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 20px 4px 12px 4px;">
                <div style="font-family:'Source Serif 4',serif;font-size:1.4rem;color:#EAF0F8;
                    letter-spacing:0.01em;">CareBot</div>
                <div style="height:1px;background:#263248;margin-top:10px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("New chat", use_container_width=True, key="new_chat_btn"):
            st.session_state.messages = []
            st.session_state.total_questions = 0
            st.session_state.pending_prompt = None
            st.session_state.awaiting_answer = False
            st.session_state.awaiting_prompt = None
            st.rerun()

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="font-size:0.72rem;color:#7C8AA0;letter-spacing:0.04em;
                margin-bottom:6px;font-family:'Inter',sans-serif;font-weight:600;">UPLOAD REPORT</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("PDF or TXT files — your documents stay private.")

        uploaded_files = st.file_uploader(
            "Choose files", type=["pdf", "txt"],
            accept_multiple_files=True, label_visibility="collapsed",
        )

        use_uploaded_files = False
        temp_vectorstore = None

        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) ready")
            with st.spinner("Processing..."):
                temp_docs = []
                temp_dir = tempfile.mkdtemp()
                for f in uploaded_files:
                    fpath = os.path.join(temp_dir, f.name)
                    with open(fpath, "wb") as out:
                        out.write(f.getbuffer())
                    try:
                        if f.name.endswith(".pdf"):
                            temp_docs.extend(PyPDFLoader(fpath).load())
                        elif f.name.endswith(".txt"):
                            temp_docs.extend(TextLoader(fpath).load())
                    except Exception as e:
                        st.warning(f"Could not read {f.name}: {e}")

                if temp_docs:
                    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    temp_vectorstore = Chroma.from_documents(temp_docs, emb, collection_name="temp_uploads")
                    use_uploaded_files = st.checkbox(
                        "Answer from uploaded docs only",
                        value=False,
                    )

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='height:1px;background:#263248;margin-bottom:16px;'></div>",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        col_a.metric("Questions", st.session_state.total_questions)
        col_b.metric("Messages", len(st.session_state.messages))

        if st.session_state.messages:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            chat_data = {
                "exported_at": datetime.now().isoformat(),
                "total_questions": st.session_state.total_questions,
                "messages": st.session_state.messages,
            }
            st.download_button(
                label="Export chat",
                data=json.dumps(chat_data, indent=2),
                file_name=f"carebot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                key="download_btn",
            )

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="height:1px;background:#263248;margin-bottom:18px;"></div>
            <div style="display:flex;flex-direction:column;gap:12px;margin-bottom:18px;">
                <div style="display:flex;align-items:center;gap:9px;">
                    <span style="color:#35D6C0;font-size:0.9rem;">✓</span>
                    <span style="font-size:0.78rem;color:#AAB8CC;font-family:'Inter',sans-serif;">Evidence-based information</span>
                </div>
                <div style="display:flex;align-items:center;gap:9px;">
                    <span style="color:#35D6C0;font-size:0.9rem;">✓</span>
                    <span style="font-size:0.78rem;color:#AAB8CC;font-family:'Inter',sans-serif;">Private document analysis</span>
                </div>
                <div style="display:flex;align-items:center;gap:9px;">
                    <span style="color:#F0B429;font-size:0.9rem;">!</span>
                    <span style="font-size:0.78rem;color:#AAB8CC;font-family:'Inter',sans-serif;">Not a replacement for professional care</span>
                </div>
            </div>
            <div style="font-size:0.7rem;color:#7C8AA0;line-height:1.6;font-family:'Inter',sans-serif;padding:0 2px;">
                For informational use only. Not a substitute for professional medical advice.
            </div>
            """,
            unsafe_allow_html=True,
        )

    has_messages = bool(st.session_state.messages)

    # ── Main header ───────────────────────────────────────────────────────
    if not has_messages:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:22px;margin-bottom:4px;margin-top:8px;">
                <div style="flex-shrink:0;">
                    <svg width="64" height="64" viewBox="0 0 72 72" fill="none">
                        <line x1="36" y1="6" x2="36" y2="16" stroke="#35D6C0" stroke-width="2.5" stroke-linecap="round"/>
                        <circle cx="36" cy="4.5" r="3" fill="#35D6C0"/>
                        <rect x="16" y="16" width="40" height="30" rx="9" fill="#121B2E" stroke="#35D6C0" stroke-width="1.8"/>
                        <rect x="22" y="24" width="10" height="7" rx="3" fill="#1B4A44"/>
                        <rect x="40" y="24" width="10" height="7" rx="3" fill="#1B4A44"/>
                        <circle cx="27" cy="27.5" r="2.5" fill="#35D6C0"/>
                        <circle cx="45" cy="27.5" r="2.5" fill="#35D6C0"/>
                        <rect x="25" y="36" width="22" height="5" rx="2.5" fill="#1B4A44"/>
                        <rect x="27" y="37.5" width="4" height="2" rx="1" fill="#35D6C0"/>
                        <rect x="33" y="37.5" width="4" height="2" rx="1" fill="#35D6C0"/>
                        <rect x="39" y="37.5" width="4" height="2" rx="1" fill="#35D6C0"/>
                        <rect x="8" y="22" width="8" height="14" rx="4" fill="#121B2E" stroke="#35D6C0" stroke-width="1.5"/>
                        <rect x="56" y="22" width="8" height="14" rx="4" fill="#121B2E" stroke="#35D6C0" stroke-width="1.5"/>
                        <rect x="20" y="48" width="32" height="18" rx="6" fill="#121B2E" stroke="#35D6C0" stroke-width="1.8"/>
                        <rect x="33" y="52" width="6" height="10" rx="1.5" fill="#35D6C0"/>
                        <rect x="30" y="55" width="12" height="4" rx="1.5" fill="#35D6C0"/>
                        <rect x="22" y="66" width="10" height="5" rx="2.5" fill="#121B2E" stroke="#35D6C0" stroke-width="1.5"/>
                        <rect x="40" y="66" width="10" height="5" rx="2.5" fill="#121B2E" stroke="#35D6C0" stroke-width="1.5"/>
                    </svg>
                </div>
                <div>
                    <div style="font-family:'Source Serif 4',serif;font-size:2.4rem;line-height:1.1;
                        color:#EAF0F8;">Ask CareBot</div>
                    <div style="font-family:'Inter',sans-serif;font-size:0.85rem;color:#7C8AA0;
                        margin-top:6px;">Your personal AI health assistant</div>
                </div>
            </div>
            <div style="height:1px;background:linear-gradient(90deg,#263248 0%,transparent 85%);margin-bottom:30px;"></div>
            """,
            unsafe_allow_html=True,
        )

    if use_uploaded_files and temp_vectorstore:
        st.info("Answering from your uploaded documents only.")
    elif temp_vectorstore:
        st.info("Searching both uploaded documents and base knowledge.")

    # ── Quick Suggestion Cards ────────────────────────────────────────────
    if not has_messages:
        st.markdown("<div class='section-label'>Common questions</div>", unsafe_allow_html=True)

        cols = st.columns(4)
        for i, (label, question) in enumerate(QUICK_SUGGESTIONS):
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="suggestion-card">
                        <div class="cat">{label}</div>
                        <div class="q">{question}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Ask", key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_prompt = question
                    st.rerun()

        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # ── Conversation history ──────────────────────────────────────────────
    # Rendered inside a real container (not just markdown text) so the
    # `.st-key-chat_messages_anchor` CSS rule above can actually target it
    # and push a short conversation down to sit right above the composer.
    chat_anchor = st.container(key="chat_messages_anchor")
    with chat_anchor:
        for message in st.session_state.messages:
            render_message(message["role"], message["content"])

        # If a question was just submitted, show it "thinking" right here in
        # the history stream (above the input bar) instead of after it.
        if st.session_state.awaiting_answer:
            with st.chat_message("assistant"):
                st.markdown(BOT_AVATAR_HTML, unsafe_allow_html=True)
                with st.spinner("CareBot is thinking..."):
                    try:
                        answer = get_answer(
                            st.session_state.awaiting_prompt, use_uploaded_files, temp_vectorstore
                        )
                    except Exception as e:
                        answer = f"Something went wrong: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.awaiting_answer = False
            st.session_state.awaiting_prompt = None
            st.rerun()

    st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <script>
            const el = document.getElementById('chat-bottom');
            if (el) el.scrollIntoView({ behavior: 'smooth' });
        </script>
        """,
        unsafe_allow_html=True,
    )

    # ── Fixed input bar ───────────────────────────────────────────────────
    # Using st.container(key=...) instead of a raw st.markdown('<div>') so the
    # CSS rules above actually scope to real DOM children (fixes both the
    # positioning and the white input background).
    input_row = st.container(key="custom_input_row")
    with input_row:
        if HAS_SPEECH_RECOGNITION:
            col_text, col_mic, col_send = st.columns([11, 1, 1])
        else:
            col_text, col_send = st.columns([12, 1])
            col_mic = None

        with col_text:
            st.text_input(
                "input",
                label_visibility="collapsed",
                placeholder="Ask a health question...",
                key="chat_text_input",
                on_change=_submit_input,
            )

        if col_mic:
            mic_class = "mic-btn-active" if st.session_state.show_voice else "mic-btn"
            st.markdown(f'<div class="{mic_class}">', unsafe_allow_html=True)
            with col_mic:
                if st.button("🎙", key="mic_toggle"):
                    st.session_state.show_voice = not st.session_state.show_voice
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        with col_send:
            if st.button("↑", key="send_btn", on_click=_submit_input):
                pass
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Voice input ───────────────────────────────────────────────────────
    if HAS_SPEECH_RECOGNITION and st.session_state.show_voice:
        audio_data = st.audio_input("Speak now", label_visibility="visible")
        if audio_data:
            with st.spinner("Converting speech to text..."):
                voice_prompt = audio_to_text(audio_data.getvalue())
                if voice_prompt:
                    st.success(f"Heard: **{voice_prompt}**")
                    st.session_state.pending_prompt = voice_prompt
                    st.session_state.show_voice = False

    # ── Resolve the final prompt ────────────────────────────────────────
    prompt = None
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    # ── Process the prompt ──────────────────────────────────────────────
    # Just record it and rerun — the actual "thinking" + answer rendering
    # happens above, inside the conversation-history section, so it always
    # appears above the input bar instead of flashing below it.
    if prompt:
        st.session_state.messages.append({"role": "User", "content": prompt})
        st.session_state.total_questions += 1
        st.session_state.awaiting_answer = True
        st.session_state.awaiting_prompt = prompt
        st.rerun()


if __name__ == "__main__":
    main()