import os
import chromadb
import google.generativeai as genai

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)
print("Gemini configured successfully")

# Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# Embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Connect ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "ml_notes"
)

while True:

    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    # Convert question to embedding
    query_embedding = embedding_model.encode(
        question
    ).tolist()

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n".join(
        results["documents"][0]
    )

    prompt = f"""
You are an AI Interview Assistant.

Answer the question only using the context below.

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(
        prompt
    )

    print("\nANSWER:\n")
    print(response.text)