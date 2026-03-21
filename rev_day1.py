# REVISION DAY 1
# Semantic Search + Manual Transformer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


# -------- PART 1: SEMANTIC SEARCH -------- #

model_embed = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Artificial intelligence is transforming healthcare.",
    "Deep learning improves computer vision.",
    "Cooking requires heat and ingredients.",
    "Airplanes fly using lift and aerodynamics."
]


doc_embeddings = model_embed.encode(documents)

query = "How do planes fly?"

query_embedding = model_embed.encode([query])

scores = cosine_similarity(query_embedding, doc_embeddings)[0]

top_index = np.argmax(scores)