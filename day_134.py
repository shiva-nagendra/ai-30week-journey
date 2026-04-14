#week 22 day 4
#FAISS with cosine similarity

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#sentences
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

#Embeddings
sen_emb = model.encode(sentences)
sen_emb = np.array(sen_emb).astype("float32")

#normalize
faiss.normalize_L2(sen_emb)

#Index
dimension = sen_emb.shape[1]
index = faiss.IndexFlatL2(dimension)


