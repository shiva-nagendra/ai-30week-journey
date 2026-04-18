#week 22 project
#Scalable semantic search using FAISS

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from data import data

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#texts
texts = [item["text"] for item in data]

#Embedings
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

#Normalize
faiss.normalize_L2(embeddings)

#Index creation
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)



