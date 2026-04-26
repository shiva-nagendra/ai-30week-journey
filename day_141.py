#week 23 day 5
#multi step retrieval RAG

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline

#models
model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="distilgpt2")

#data
documents = [
    "AI helps diagnose diseases",
    "Machine learning analyzes patient data",
    "Deep learning improves medical imaging",
    "AI is used in drug discovery",
    "Doctors use AI for treatment planning"
]

doc_emb = model.encode(documents)

query = input("Enter your question: ")

#Step 1: intial Retrieve 

query_emb = model.encode([query])
scores = cosine_similarity(query_emb, doc_emb)[0]
top_indices = np.argsort(scores)[::-1][:5]


