#week 21 day 4
#Vector database with FAISS

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#Documents
documents = [
    "AI helps healthcare",
    "Machine learning analyzes data",
    "Rockets go to space",
    "Deep learning improves vision",
    "Cooking requires ingredients"
]

#Convert to embeddings
doc_embeddings = model.encode(documents)


