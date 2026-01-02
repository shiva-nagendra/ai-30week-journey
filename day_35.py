#Turn text into vectors and compare meaning using distance.
#local

from sentence_transformers import SentenceTransformer
import math

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love trekking and sports",
    "sports are good for health",
    "politicians should be sportive in nature",
    "natural growth in sports should be encouraged"
]

embeddings = model.encode(sentences)

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    return dot / (norm_a * norm_b)

base = embeddings[0]

for i, emb in enumerate(embeddings):
    sim = cosine_similarity(base, emb)
    print(f"Similarity sentence 0 vs {i}: {round(sim,3)}")

