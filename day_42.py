#Mini End-to-End AI System

# User Query
#    ↓
# Embedding
#    ↓
# Semantic Retrieval (memory)
#    ↓
# Constraints / Evaluation
#    ↓
# Decision
#    ↓
# Grounded Answer

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

def answer_query(query):
    #EMBEDDING QUERRY
    query_embedding  = model.encode([query])[0]

    #RETRIEVE

    scores = []
    for text, emb in zip (memory, memory_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((text,float(sim)))

    scores.sort(key=lambda x: x[1], reverse=True)

    print("\nTop retrieval results:")
    for text, score in scores[:3]:
        print(f"{round(score,3)} -> {text}")


    #DECISION CONSTRAINTS
    if scores[0][1] >= 0.6 and scores[1][1] >= 0.5:
        decision = "ANSWER_USING_MEMORY"
    else:
        decision = "INSUFFICIENT CONTEXT"

    #response 

    if decision == "ANSWER_USING_MEMORY":
        response="Answering based on what I know.\n"
        for text, score in scores[:2]:
            response += f"- {text}\n"
    else:
        response = "Insufficient information"
    
    return decision, response

if __name__ == "__main__":
    decision, response = answer_query("model embeddings") 
    print("\nDecision:",decision)
    print("\nResponse:",response)




