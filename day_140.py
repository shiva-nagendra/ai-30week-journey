#week 23 day 4
#Chunkning ranking for RAG

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np      
from transformers import pipeline

# Models

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="distilgpt2")

# Data

documents = [

    "AI helps diagnose diseases",

    "Machine learning analyzes patient data",

    "Deep learning improves medical imaging",

    "AI is used in drug discovery",

    "Doctors use AI for treatment planning"

]

doc_embeddings = embed_model.encode(documents)

# -------- QUERY -------- #

query = input("Enter your question: ")
query_emb = embed_model.encode([query])

