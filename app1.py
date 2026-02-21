import os
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load PDF
reader = PdfReader("sample.pdf")
full_text = ""

for page in reader.pages:
    full_text += page.extract_text()

# 🔥 Split into chunks
chunk_size = 800
chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

# Ask question
question = input("Ask a question about the PDF: ").lower()

# 🔥 Simple keyword matching retrieval
relevant_chunks = []

for chunk in chunks:
    if any(word in chunk.lower() for word in question.split()):
        relevant_chunks.append(chunk)

# If nothing matched, fallback to first chunk
if not relevant_chunks:
    relevant_chunks = chunks[:2]

context = "\n".join(relevant_chunks[:3])

# Send only relevant context
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

print("\nAnswer:")
print(response.choices[0].message.content)
import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()

# Load embedding model (local, free)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load PDF
reader = PdfReader("sample.pdf")
full_text = ""

for page in reader.pages:
    full_text += page.extract_text()

# Split into chunks
chunk_size = 800
chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

# Convert chunks to embeddings
chunk_embeddings = embedding_model.encode(chunks)

# Convert to numpy array (required by FAISS)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")

# Create FAISS index
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

print("✅ PDF embedded and stored in FAISS successfully!")
print(f"Total chunks stored: {len(chunks)}")
