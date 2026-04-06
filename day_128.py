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

#Convert to numpy float32 
doc_embeddings = np.array(doc_embeddings).astype("float32")

#Create faiss index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

#Add embeddings to index
index.add(doc_embeddings)


