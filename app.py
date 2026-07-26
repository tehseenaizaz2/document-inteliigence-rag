import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="DocuMind RAG", page_icon="📄", layout="wide")

st.title("📄 DocuMind Intelligence Hub")
st.subheader("Grounded Document Q&A powered by RAG & Groq")

# 1. Initialize Engine & Indexing State in Session State
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# 2. Sidebar for PDF Upload & Indexing
with st.sidebar:
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Index Document", type="primary"):
            with st.spinner("Extracting & Indexing Text..."):
                # Reset engine instance on new upload to clear previous collection state
                st.session_state.engine = RAGEngine()
                num_chunks = st.session_state.engine.index_document(uploaded_file, uploaded_file.name)
                
                if num_chunks > 0:
                    st.session_state.indexed = True
                    st.success(f"✅ Successfully indexed {num_chunks} text chunks!")
                else:
                    st.session_state.indexed = False
                    st.error("❌ No text could be extracted! If this is a scanned/image PDF, please try a text-based PDF.")

# 3. Main Q&A Interface
user_query = st.text_input("Ask a question about your uploaded document:")

if user_query:
    if not st.session_state.indexed:
        st.warning("⚠️ Please upload a PDF and click 'Index Document' in the sidebar first!")
    else:
        with st.spinner("Searching document & generating answer..."):
            retrieved_chunks = st.session_state.engine.retrieve(user_query, top_k=4)
            answer = st.session_state.engine.generate_answer(user_query, retrieved_chunks)
            
            st.markdown("### 🤖 Answer:")
            st.write(answer)
            
            if retrieved_chunks:
                with st.expander("📚 View Retrieved Source Passages"):
                    for idx, chunk in enumerate(retrieved_chunks, 1):
                        st.write(f"**Source {idx} (Page {chunk['page']}):**")
                        st.caption(chunk['text'])
                        st.divider()