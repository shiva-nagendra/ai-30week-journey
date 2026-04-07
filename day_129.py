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

print("\nchunks:\n")
for c in chunks:
    print("-", c)

