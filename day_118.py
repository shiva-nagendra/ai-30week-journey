#week 19 day 6
#Mini semantic search engine

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

#knowledge base
