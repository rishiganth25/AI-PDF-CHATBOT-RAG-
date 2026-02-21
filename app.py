import os
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# Create Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load PDF
reader = PdfReader("Rishi Ganth.s_Resume.pdf")

full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

# Ask user question
question = input("Ask a question about the PDF: ")

# Send PDF content + question to AI
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer only from the given PDF content."
        },
        {
            "role": "user",
            "content": f"Here is the PDF content:\n{full_text}\n\nQuestion: {question}"
        }
    ]
)

print("\nAnswer:")
print(response.choices[0].message.content)
