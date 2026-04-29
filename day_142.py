#week 23 day 6
#RAG evaluation

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#model
model = SentenceTransformer("all-MiniLM-L6-v2")

#data
documents = [
    "AI helps diagnose diseases",
    "Machine learning analyzes patient data",
    "Deep learning improves medical imaging",
    "AI is used in drug discovery",
    "Doctors use AI for treatment planning"
]

