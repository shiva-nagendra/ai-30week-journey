#week 22 day 6
# Top-K tuning and reranking

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Data
data = [
    {"text": "AI helps diagnose diseases", "category": "health"},
    {"text": "Machine learning analyzes data", "category": "tech"},
    {"text": "Deep learning improves imaging", "category": "health"},
    {"text": "Space rockets explore planets", "category": "space"},
    {"text": "Healthy food improves life", "category": "health"},
    {"text": "Solar energy is renewable", "category": "energy"},
]

texts = [item["text"] for item in data]

# Embeddings
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

faiss.normalize_L2(embeddings)

#Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

#query
query = input("\nEnter your query: ")
query_emb = model.encode(query)
faiss.normalize_L2(query_emb)

#Retrive more
k = 5
distances, indices = index.search(query_emb, k)

#Filter
filtered = []

for i, idx in enumerate(indices[0]):
    item = data[idx]
    score = 1 - distances[0][i]
    filtered.append((item["text"], item["category"], score))

top_results = filtered[:3]









