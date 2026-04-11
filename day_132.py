#week 22 day 2
#FAISS implementation

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Data
sentences = [
    "AI is transforming healthcare",
    "Machine learning analyzes data",
    "Deep learning improves vision",
    "Space rockets explore planets",
    "Healthy food improves life",
    "Exercise keeps the body fit",
    "Solar energy is renewable",
    "Wind power generates electricity"
]

