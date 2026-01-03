# Query
#  ↓
# Retrieve relevant info
#  ↓
# Use retrieved info as a SIGNAL
#  ↓
# Decision / Response

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
    return dot / (norm_a*norm_b)

query = "How does exercise affect your mind?"
query_embeddings = model.encode([query])[0]

scores = []

for text, emb in zip(memory, memory_embeddings):
    sim = cosine_similarity(query_embeddings, emb)
    scores.append((text, sim))

scores.sort(key=lambda x: x[1], reverse= True)


for item, score in scores[:1]:
    print(f"{round(score,3),}-->{item}")


TOP_SCORE_THRESHOLD =  0.4 #Retrival into signal

top_item, top_score = scores[0]

if top_score >= TOP_SCORE_THRESHOLD:
    knowledge_found = True
else:
    knowledge_found = False

if knowledge_found:
    decision = "ANSWER USING MEMORY"
else:
    decision = "INSUFFICIENT CONTEXT"

print("Decision: ", decision)




