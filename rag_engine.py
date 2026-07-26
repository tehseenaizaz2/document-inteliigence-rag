"""
rag_engine.py
--------------
Core RAG logic: PDF parsing, chunking, vector storage (ChromaDB),
semantic retrieval, and answer generation using Groq API with source citations.
"""

import chromadb
from chromadb.api.types import EmbeddingFunction
from pypdf import PdfReader
import uuid
import re
import math
import os
from dotenv import load_dotenv  
from groq import Groq

# Load environment variables from .env file
load_dotenv()


class LightweightEmbeddingFunction(EmbeddingFunction):
    """
    A fast, no-download embedding function based on hashed word frequencies
    (feature hashing). Avoids downloading large ML models over slow or
    unstable internet connections. Good enough for small-to-medium document
    Q&A demos and portfolio projects.
    """

    def __init__(self, dim=384):
        self.dim = dim

    def __call__(self, input):
        return [self._embed(text) for text in input]

    def _embed(self, text):
        vec = [0.0] * self.dim
        words = re.findall(r"\w+", text.lower())
        for word in words:
            idx = hash(word) % self.dim
            vec[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class RAGEngine:
    def __init__(self, collection_name="documents"):
        # Initialize Groq client (Reads GROQ_API_KEY from environment)
        api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        
        self.chroma_client = chromadb.Client()  # in-memory vector store
        self.embedding_fn = LightweightEmbeddingFunction()

        # Reset collection each time a new session starts
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def extract_text_from_pdf(self, file):
        reader = PdfReader(file)
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages_text.append((i + 1, text))
        return pages_text

    def chunk_text(self, pages_text, chunk_size=600, overlap=100):
        """Split each page's text into overlapping chunks, tagged with page number."""
        chunks = []
        for page_num, text in pages_text:
            start = 0
            while start < len(text):
                chunk = text[start:start + chunk_size].strip()
                if chunk:
                    chunks.append({"text": chunk, "page": page_num})
                start += chunk_size - overlap
        return chunks

    def index_document(self, file, source_name="uploaded_document"):
        """Extract, chunk, and store a PDF's content as vector embeddings."""
        pages_text = self.extract_text_from_pdf(file)
        chunks = self.chunk_text(pages_text)

        if not chunks:
            return 0

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{"page": c["page"], "source": source_name} for c in chunks]

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return len(chunks)

    def retrieve(self, question, top_k=4):
        """Semantic search: find the most relevant chunks for a question."""
        results = self.collection.query(query_texts=[question], n_results=top_k)

        retrieved = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved.append({"text": doc, "page": meta["page"], "source": meta["source"]})
        return retrieved

    def generate_answer(self, question, retrieved_chunks):
        """Answer using only retrieved context via Groq LLM (Llama 3.1 8B)."""
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_blocks.append(f"[Source {i}, page {chunk['page']}]\n{chunk['text']}")

        context = "\n\n".join(context_blocks)

        prompt = f"""Answer the question using ONLY the context below. Cite sources
using the format [Source N] after each claim. If the answer isn't in the context,
say you don't have enough information.

CONTEXT:
{context}

QUESTION:
{question}
"""
        # Updated to Groq Chat Completion standard
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700
        )
        
        return response.choices[0].message.content