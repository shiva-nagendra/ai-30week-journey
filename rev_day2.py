# REVISION DAY 2
# Advanced Semantic Search + Manual Transformer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# -------- PART 1: SEMANTIC SEARCH (TOP-K) -------- #

model_embed = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Artificial intelligence is transforming healthcare.",
    "Machine learning helps analyze data.",
    "Deep learning improves image recognition.",
    "Airplanes fly using lift and aerodynamics.",
    "Cooking requires proper ingredients and heat.",
    "Programming builds software systems."
]

doc_embeddings = model_embed.encode(documents)

#user query
query = input("Enter your query: ")

query_embedding = model_embed.encode([query])

scores = cosine_similarity(query_embedding, doc_embeddings)[0]

