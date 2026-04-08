#week 21 day 6
#context building for RAG

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")

