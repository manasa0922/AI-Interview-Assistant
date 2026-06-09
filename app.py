import os
import streamlit as st
import chromadb
import google.generativeai as genai

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load API Key
load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Gemini Model
model = genai.GenerativeModel("gemini-2.5-flash")

# Embedding Model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "ml_notes"
)

# UI
st.title("🤖 AI Interview Assistant")

question = st.text_input(
    "Ask a question:"
)

if st.button("Ask"):
 with st.spinner("Searching and generating answer..."):
    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n".join(
        results["documents"][0]
    )

    prompt = f"""
Answer the question only using the context below.

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(
        prompt
    )

    st.subheader("Answer")

    st.write(response.text)