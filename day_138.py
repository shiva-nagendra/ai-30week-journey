#week 23 day 2
#Retrieval + Generation

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

#Embedding model
emb_model = SentenceTransformer("all-MiniLM-L6-v2")

#Generation model
model_generator = pipeline("text-generation", model="distilgpt2")

#Data
documents = [
    "AI helps diagnose diseases",
    "Machine learning analyzes patient data",
    "Deep learning improves medical imaging",
    "AI is used in drug discovery",
    "Doctors use AI for treatment planning"
]

doc_emb = emb_model.encode(documents)

query = input("Enter your query: ")

query_emb = emb_model.encode([query])

