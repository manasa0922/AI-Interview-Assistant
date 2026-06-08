from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Machine Learning is a subset of AI",
    "Artificial Intelligence allows machines to learn"
]

embeddings = model.encode(sentences)

print("Embedding Length:", len(embeddings[0]))

print("\nFirst 10 Values:")
print(embeddings[0][:10])