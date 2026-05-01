#week 23 project
#RAG - Retrieval Augmented Generation

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data import data

# Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation",model="distilgpt2")

#Data
texts = [item["text"] for item in data]
doc_emb = embed_model.encode(texts)

#Main loop
while True:
    query = input("\nEnter your querry: (or exit)")
    if query == "exit".lower():
        break

    expanded_queries = [

        query,
        query + "in healthcare.",
        query + "medical applications."
        
    ]
    
    all_scores = []

    for q in expanded_queries:




