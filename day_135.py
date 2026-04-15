#week 22 day 5
#metadata filtering with FAISS

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#doc
data = [
     {"text": "AI helps diagnose diseases", "category": "health"},
    {"text": "Machine learning analyzes data", "category": "tech"},
    {"text": "Deep learning improves imaging", "category": "health"},
    {"text": "Space rockets explore planets", "category": "space"},
    {"text": "Healthy food improves life", "category": "health"},
    {"text": "Solar energy is renewable", "category": "energy"},
]

texts = [item["text"] for item in data]

#Embeddings
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

faiss.normalize_L2(embeddings)

#index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

#Query
query = input("\nEnter your query: ")
category_filter = input("\nEnter category (or 'all'):")

query_emb = model.encode(query).astype("float32")
faiss.normalize_L2(query_emb)

#Search
k = 5
distances, indices = index.search(query_emb, k)

print("\nFiltered results: ")



