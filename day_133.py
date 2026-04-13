#week 22 day 3
#FAISS IVF SEARCH

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#Data
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
embeddings = model.encode(sentences)
embeddings = np.array(embeddings).astype("float32")

#Dimension
dimension = embeddings.shape[1]

#IVF setup
nlist = 2 #No. of clusters

quantizer = faiss.IndexFlatL2(dimension)#base index

#Train index
index = faiss.IndexIVFFlat(quantizer,dimension,nlist)

#Add vectors
index.add(embeddings)

#query

query = input("\nEnter your query: ")
query_embedding = model.encode([query]).astype("float32")

