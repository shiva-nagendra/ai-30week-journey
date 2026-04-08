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


