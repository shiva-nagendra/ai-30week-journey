#week 19 project
# Semantic search engine(Top-k retrieval)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#Knowledge base

