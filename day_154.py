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
    print(f"\nIntial score: {scores[idx]:.3f}")

#Rerank scores
rerank_scores = []

query_words = query.lower().split()

for idx in top_indices:
    doc = documents[idx].lower()
    overlap_score = 0

    for word in query_words:
        if word in doc:
            overlap_score += 1

    final_score = (
        scores[idx] * 0.7
        +
        overlap_score * 0.3
    )
    rerank_scores.append((idx, final_score))

#sort reranked results
rerank_scores.sort(
    key=lambda x: x[1],
    reverse=True
)

#Final output

print("\nRERANKED RESULTS:\n")
for idx, score in rerank_scores:

    print(documents[idx])

    print(f"Reranked Score: {score:.3f}\n")

