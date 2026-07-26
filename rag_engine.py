import os
import chromadb
from pypdf import PdfReader
from groq import Groq

class RAGEngine:
    def __init__(self):
        # Read API key from environment variable (set via Streamlit Secrets)
        self.api_key = os.environ.get("GROQ_API_KEY")
        if self.api_key:
            self.groq_client = Groq(api_key=self.api_key)
        else:
            self.groq_client = None
        
        # Initialize Ephemeral ChromaDB Client
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_documents"
        )

    def index_document(self, file_obj, source_name="uploaded_pdf"):
        """Extracts text from PDF, chunks it, and indexes into ChromaDB."""
        try:
            reader = PdfReader(file_obj)
            documents = []
            metadatas = []
            ids = []
            
            chunk_id = 0
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if not text or not text.strip():
                    continue
                
                paragraphs = text.split("\n\n")
                for para in paragraphs:
                    clean_para = para.strip()
                    if len(clean_para) > 30:
                        documents.append(clean_para)
                        metadatas.append({
                            "source": source_name,
                            "page": page_num
                        })
                        ids.append(f"{source_name}_p{page_num}_c{chunk_id}")
                        chunk_id += 1

            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            return len(documents)
            
        except Exception as e:
            print(f"Error indexing document: {e}")
            return 0

    def retrieve(self, question, top_k=4):
        """Safely queries vector collection."""
        try:
            if self.collection.count() == 0:
                return []
                
            results = self.collection.query(
                query_texts=[question],
                n_results=top_k
            )
            
            retrieved_chunks = []
            if results and 'documents' in results and results['documents']:
                docs = results['documents'][0]
                metas = results['metadatas'][0] if 'metadatas' in results else []
                
                for i in range(len(docs)):
                    retrieved_chunks.append({
                        "text": docs[i],
                        "source": metas[i].get("source", "Unknown") if i < len(metas) else "Unknown",
                        "page": metas[i].get("page", 1) if i < len(metas) else 1
                    })
            return retrieved_chunks
            
        except Exception as e:
            print(f"ChromaDB Query Error: {e}")
            return []

    def generate_answer(self, question, context_chunks):
        """Generates grounded answer using Groq Llama 3.1 model."""
        if not context_chunks:
            return "I couldn't find any relevant information in the uploaded documents to answer your question."

        # Re-check API key in case it was updated dynamically
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return "⚠️ Groq API Key is missing. Please add GROQ_API_KEY in Streamlit App Secrets."

        client = self.groq_client or Groq(api_key=api_key)

        context_str = ""
        for i, chunk in enumerate(context_chunks, start=1):
            context_str += f"[Source {i} - {chunk['source']}, Page {chunk['page']}]:\n{chunk['text']}\n\n"

        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question accurately using ONLY "
            "the provided document excerpts. If the information is not present in the excerpts, "
            "state clearly that the document doesn't mention it."
        )

        user_prompt = f"Context excerpts:\n{context_str}\nQuestion: {question}"

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=700
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ Groq API Error: {str(e)}. Please verify your API Key in Streamlit Cloud Secrets."