#week 25 day 6
#Ranking restrieved results

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
     "AI helps doctors diagnose diseases",

    "Machine learning improves hospital workflow",

    "AI predicts stock market trends",

    "Deep learning improves medical imaging",

    "Hospitals use AI for patient care"
]

#Model
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_emb = model.encode(documents)

#Query
query = input("Enter your query: ")
query_emb = model.encode([query])



