#week 21 day 5
#Chunking for RAG

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#Large document
document = """
Artificial intelligence is transforming healthcare.
It helps doctors diagnose diseases faster.
Machine learning models analyze patient data.
Deep learning improves medical imaging systems.
AI also helps in drug discovery.
"""

chunks = document.strip().split("\n")

chunk_emb = model.encode(chunks)

print("\nchunks:\n")
for c in chunks:
    print("-", c)

querry = input("\nEnter your querry:")

querry_emb = model.encode([querry])

scores = cosine_similarity(querry_emb, chunk_emb)[0]
