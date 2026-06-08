import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("ml_notes")

# User Question
query = "Explain supervised learning"
# Convert question to embedding
query_embedding = model.encode(query).tolist()

# Search
# Search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print("\nQUESTION:")
print(query)

print("\nMOST RELEVANT CHUNKS:\n")

for doc in results["documents"][0]:
    print(doc)
    print("\n" + "-" * 50 + "\n")