#week 21 day 6
#context building for RAG

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#Documents
document = """
AI helps diagnose diseases.
Machine learning analyzes patient data.
Deep learning improves medical imaging.
AI is also used in drug discovery.
"""

Chunks = document.strip().split()("\n")

chunk_emb = model.encode(Chunks)

querry = input("\nEnter your querry: ")

querry_emb = model.encode([querry])

scores =cosine_similarity(querry_emb, chunk_emb)[0]

top_indices = np.argsort(scores)[::-1][:2]

context = " ".join([Chunks[idx] for idx in top_indices])

print("\nRetrieved context:")
print(context)

print("\nGenerated answer:")
print(f"\nAnswer: {querry} -> {context}")


