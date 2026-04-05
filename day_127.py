#week 21 day 3
#Semantic search engine

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#Knowledge base (documents)
sentences = [
    "Artificial intelligence is transforming healthcare.",
    "Machine learning helps analyze large datasets",
    "Space rockets travel beyond earth's atmosphere.",
    "Deep learning improves computer vision system",
    "Cooking pasta requires boiling water"
]

#convert dox to embeddings
doc_emb = model.encode(sentences)

#User querry
querry = input("Enter your querry")

