# Day 41 — model working check

from sentence_transformers import SentenceTransformer
import math


memory = [
    "Regular exercise improves physical health",
    "Sports help maintain mental well-being",
    "AI models work on numerical representations",
    "AI models are replacing search engines.",
    "Embeddings capture semantic meaning of text",
    "Politics influences public policy",
    "Policies are the rules for any system"
]

model = SentenceTransformer("all-MiniLM-L6-v2")
memory_embeddings = model.encode(memory)

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

query = "Artificial intelligence. and AI " #testing the relavnce of the querry
query_embedding = model.encode([query])[0]

scores = []
for text, emb in zip(memory, memory_embeddings):
    similarity = cosine_similarity(query_embedding, emb)
    scores.append((text, float(similarity)))

scores.sort(key=lambda x: x[1], reverse=True)

print("\nTop Retrieval Results:")
for text, score in scores[:3]:
    print(f"{round(score, 3)} -> {text}")


PRIMARY_THRESHOLD = 0.6
SECONDARY_THRESHOLD = 0.5

top_score = scores[0][1]
support_score = scores[1][1]

knowledge_found = (
    top_score >= PRIMARY_THRESHOLD
    and support_score >= SECONDARY_THRESHOLD
)

decision = "ANSWER_USING_MEMORY" if knowledge_found else "INSUFFICIENT_CONTEXT"
print("\nDecision:", decision)

if decision == "ANSWER_USING_MEMORY":
    answer = (
        "Answer is based on the retrieved information."
    )
else:
    answer = "I don’t have enough information to answer that confidently."

print("\nFinal Answer:")
print(answer)