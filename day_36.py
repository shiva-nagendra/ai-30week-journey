# User input.   #RAG
#    ↓
# Embedding
#    ↓
# Search memory (semantic similarity)
#    ↓
# Relevant information
#    ↓
# Decision / response

memory = [
    "Regular exercise improves physical health",
    "Sports help maintain mental well-being",
    "AI models work on numerical representations",
    "Embeddings capture semantic meaning of text",
    "Politics influences public policy",
    "Policies are the rules for any system"
]

from sentence_transformers import SentenceTransformer
import math

model = SentenceTransformer("all-MiniLM-L6-v2")
memory_embeddings = model.encode(memory)

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    return dot/(norm_a*norm_b)

query = "what is the role of politics?"
query_embedding = model.encode([query])[0]

scores = []

for text, emb in zip(memory, memory_embeddings):
    sim = cosine_similarity(query_embedding, emb)
    scores.append((text, sim))

scores.sort(key= lambda x: x[1], reverse=True)

for item, score in scores[:3]:
    print(f"{round(score,3),}-->{item}")     