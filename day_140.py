#week 23 day 4
#Chunkning ranking for RAG

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np      
from transformers import pipeline

# Models

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

generator = pipeline("text-generation", model="distilgpt2")

