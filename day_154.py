#week 25 day 6
#Ranking restrieved results

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
     "AI helps doctors diagnose diseases",

    "Machine learning improves hospital workflow",

    "AI predicts stock market trends",

    "Deep learning improves medical imaging",

    "Hospitals use AI for patient care"
]

#Model
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_emb = model.encode(documents)

#Query
query = input("Enter your query: ")
query_emb = model.encode([query])

#Intial retrieval
scores = cosine_similarity(
    query_emb,doc_emb
)[0]

top_indices = np.argsort(scores)[::-1][:3]

print("\nIntial retrieval:\n")

for idx in top_indices:
    print(documents[idx])
    print(f"Intial score: {scores[idx]:.3f}")

    


