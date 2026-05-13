#week 25 day 5
#Hybrid Search (Keyword + Semantic)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#docs
documents = [

    "GPT-4 Turbo supports large context windows",

    "Machine learning improves healthcare systems",

    "Python is popular for AI engineering",

    "GPT-4 pricing depends on token usage",

    "Hospitals use AI for medical imaging"

]

#embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)

#query
query = input("Ask question: ")

query_embedding = model.encode([query])

#semantic search

semantic_scores = cosine_similarity(
    query_embedding,
    doc_embeddings
)[0]

