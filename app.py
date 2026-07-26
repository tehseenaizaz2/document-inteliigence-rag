import streamlit as st
import os
from dotenv import load_dotenv
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="DocuMind AI | Smart Document QA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Vibrant CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Header Banner Styling */
    .hero-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.3);
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #E0E7FF;
        opacity: 0.9;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* File Status Badge */
    .file-badge {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1D4ED8;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }

    /* Question Box Header */
    .question-badge {
        background-color: #EEF2FF;
        color: #4338CA;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px 8px 0px 0px;
        border-left: 4px solid #6366F1;
        font-size: 1.05rem;
    }

    /* Answer Card Styling */
    .answer-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E7FF;
        border-top: none;
        border-radius: 0px 0px 12px 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        color: #1E293B;
        font-size: 0.98rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }

    /* Source Citation Cards */
    .source-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #10B981; /* Emerald Green */
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.6rem;
        font-size: 0.88rem;
        color: #334155;
    }
    .source-tag {
        font-weight: 700;
        color: #059669;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }

    /* Submit Button Custom Styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%);
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "history" not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR: Document Control Center ---
with st.sidebar:
    st.markdown("<h2 style='color:#1E293B; font-size:1.3rem; font-weight:700;'>📁 Document Workspace</h2>", unsafe_allow_html=True)
    st.caption("Upload PDFs to index them into ChromaDB vector storage.")
    st.write("")

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
        if new_files:
            with st.spinner("Embedding PDF content..."):
                for f in new_files:
                    st.session_state.engine.index_document(f, source_name=f.name)
                    st.session_state.indexed_files.append(f.name)
            st.success(f"Successfully indexed {len(new_files)} document(s)!")

    st.divider()

    # Active Files Panel
    st.markdown("<h4 style='color:#475569; font-size:0.95rem; font-weight:700;'>ACTIVE INDEX</h4>", unsafe_allow_html=True)
    if st.session_state.indexed_files:
        for file in st.session_state.indexed_files:
            st.markdown(f'<div class="file-badge">📄 &nbsp; {file}</div>', unsafe_allow_html=True)
    else:
        st.info("No documents active right now.")

    st.write("")
    if st.button("Reset Knowledge Base", use_container_width=True):
        st.session_state.engine = RAGEngine()
        st.session_state.indexed_files = []
        st.session_state.history = []
        st.rerun()

# --- MAIN HERO HEADER ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">DocuMind Intelligence Hub ⚡</div>
    <div class="hero-subtitle">High-speed vector search and grounded Q&A powered by Groq & Llama 3.1</div>
</div>
""", unsafe_allow_html=True)

# Query Input Section
with st.form(key="search_form", clear_on_submit=False):
    user_query = st.text_input(
        "Ask a question:",
        placeholder="e.g., Explain the core methodology mentioned in the document...",
        key="query_input"
    )
    submit_button = st.form_submit_button(label="Search & Analyze", type="primary", use_container_width=True)

# Query Handling & Processing
if submit_button and user_query:
    if not st.session_state.indexed_files:
        st.warning("⚠️ Please upload at least one PDF in the sidebar before asking questions.")
    else:
        with st.spinner("Analyzing document context..."):
            retrieved_chunks = st.session_state.engine.retrieve(user_query, top_k=4)
            answer = st.session_state.engine.generate_answer(user_query, retrieved_chunks)
            
            # Prepend into conversation history
            st.session_state.history.insert(0, {
                "query": user_query,
                "answer": answer,
                "sources": retrieved_chunks
            })

# Answer Display Card List
if st.session_state.history:
    st.write("")
    for item in st.session_state.history:
        # Question Bar (Indigo Accent)
        st.markdown(f"""
        <div class="question-badge">
            💡 Question: {item['query']}
        </div>
        """, unsafe_allow_html=True)
        
        # Answer Container (White Card with subtle shadow)
        st.markdown(f"""
        <div class="answer-card">
            {item['answer']}
        </div>
        """, unsafe_allow_html=True)

        # Source Citations Accordion (Emerald Accent)
        with st.expander("📌 View Verified Document Citations"):
            for i, chunk in enumerate(item["sources"], 1):
                st.markdown(f"""
                <div class="source-box">
                    <div class="source-tag">CITATION {i} — {chunk['source']} (Page {chunk['page']})</div>
                    "{chunk['text']}"
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)