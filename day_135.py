#week 22 day 5
#metadata filtering with FAISS

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#doc
sentences = [
     {"text": "AI helps diagnose diseases", "category": "health"},
    {"text": "Machine learning analyzes data", "category": "tech"},
    {"text": "Deep learning improves imaging", "category": "health"},
    {"text": "Space rockets explore planets", "category": "space"},
    {"text": "Healthy food improves life", "category": "health"},
    {"text": "Solar energy is renewable", "category": "energy"},
]

