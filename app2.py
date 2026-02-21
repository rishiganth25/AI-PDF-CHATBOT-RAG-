import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# Load environment variables
load_dotenv()

# Initialize Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("📄 AI Powered PDF Chatbot (Real RAG)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # Read PDF
    reader = PdfReader(uploaded_file)
    full_text = ""

    for page in reader.pages:
        full_text += page.extract_text()

    # Split into chunks
    chunk_size = 800
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    # Create embeddings
    chunk_embeddings = embedding_model.encode(chunks)
    chunk_embeddings = np.array(chunk_embeddings).astype("float32")

    # Create FAISS index
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)

    st.success(f"PDF processed successfully! Total chunks: {len(chunks)}")

    question = st.text_input("Ask a question about the PDF")

    if question:

        # Convert question to embedding
        question_embedding = embedding_model.encode([question])
        question_embedding = np.array(question_embedding).astype("float32")

        # Search similar chunks
        k = 3
        distances, indices = index.search(question_embedding, k)

        relevant_chunks = [chunks[i] for i in indices[0]]
        context = "\n".join(relevant_chunks)

        # Ask Groq
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Answer only using the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
        )

        st.subheader("Answer:")
        st.write(response.choices[0].message.content)
