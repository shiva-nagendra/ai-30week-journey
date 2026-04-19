#week 22 project
#Scalable semantic search using FAISS

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from data import data

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#texts
texts = [item["text"] for item in data]

#Embedings
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

#Normalize
faiss.normalize_L2(embeddings)

# FAISS Index creation
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

#Query loop

while True:
    query = input("\nEnter your query: (or 'exit')")
    if query == "exit":
        break

    #Category filter
    category_filter = input("Enter category (or 'all'): ")

    query_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)

    #Retrieve more
    k = 6
    distance, indices = index.search(query_emb, k)

    #filter + rerank
    results = []

    for i, idx in enumerate(indices[0]):
        item = data[idx]

        if category_filter == "all" or item["category"] == category_filter:
            score = 1 - distance[0][i]
            results.append((item["text"], item["category"], score))

    results = sorted(results, key=lambda x: x[2], reverse=True)

    print("\nResults:\n")

    for text, cat, score in results:
       print(f"{text} [{cat}] (score: {score:.3f})")






    



