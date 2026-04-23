#week 23 day 3
#Query improvement + RAG

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

#Embedding model
emb_model = SentenceTransformer("all-MiniLM-L6-v2")
#Generation model
generator = pipeline("text-generation", model="distilgpt2")


#data
documents = [

    "AI helps diagnose diseases",

    "Machine learning analyzes patient data",

    "Deep learning improves medical imaging",

    "AI is used in drug discovery",

    "Doctors use AI for treatment planning"

]

