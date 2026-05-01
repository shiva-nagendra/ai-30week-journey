#week 23 project
#RAG - Retrieval Augmented Generation

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data import data

# Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline()