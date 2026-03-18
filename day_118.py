#week 19 day 6
#Mini semantic search engine

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

#knowledge base
documents = [
    "Artificial intelligence is transforming healthcare.",
    "Space rockets travel beyond earth's atmosphere.",
    "Deep learning improves computer vision systems.",
    "Cooking pasta requires boiling water."
]

#convert docs into embeddings
doc_embeddings = model.encode(documents)

#user querry
querry = "How does AI help medicine?"

#Convert querry into embeddings
Query_embedding = model.encode([querry])

#Compute similarity
scores = cosine_similarity(Query_embedding,doc_embeddings)


