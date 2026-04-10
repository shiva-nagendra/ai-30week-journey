#week 22 day 1
#vector databases

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#sample data
sentences = [
    "AI is transforming healthcare",
    "Machine learning analyzes data",
    "Deep learning improves vision",
    "Space rockets explore planets",
    "Healthy food improves life",
    "Exercise keeps the body fit",
    "Solar energy is renewable",
    "Wind power generates electricity"
]

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#embeded sentences
sen_emb = model.encode(sentences)

#querry
querry = input("Enter your querry:")

query_emb = model.encode([querry])




