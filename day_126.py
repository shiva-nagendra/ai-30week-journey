#week 21 day 2
#cosine similarit between sentences

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#sentences
sentences = [
    "I love AI",
    "Artificial intelligence is amazing",
    "The weather is very hot",
    "I enjoy machine learning"
]

