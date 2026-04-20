#week 23 day 2
#Retrieval + Generation

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

#Embedding model
emb_model = SentenceTransformer("all-MiniLM-L6-v2")

#Generation model
model = pipeline("text-generation", model="distilgpt2")


