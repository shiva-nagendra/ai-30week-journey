#week 22 day 6
# Top-K tuning and reranking

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Data
data = [
    {"text": "AI helps diagnose diseases", "category": "health"},
    {"text": "Machine learning analyzes data", "category": "tech"},
    {"text": "Deep learning improves imaging", "category": "health"},
    {"text": "Space rockets explore planets", "category": "space"},
    {"text": "Healthy food improves life", "category": "health"},
    {"text": "Solar energy is renewable", "category": "energy"},
]

texts = [item["text"] for item in data]

# Embeddings
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")


