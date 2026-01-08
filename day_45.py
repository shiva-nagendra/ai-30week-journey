#stress testing

from sentence_transformers import SentenceTransformer
import math

_MEMORY = [
    "Regular exercise improves physical health",
    "Sports help maintain mental well-being",
    "AI models work on numerical representations",
    "Embeddings capture semantic meaning of text",
    "Politics influences public policy",
    "Policies are the rules for any system"
]

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_MEMORY_EMBEDDINGS = _MODEL.encode(_MEMORY)

def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def _retrieve(query_embedding):
    scores = []
    for text, emb in zip(_MEMORY, _MEMORY_EMBEDDINGS):
        sim = _cosine_similarity(query_embedding, emb)
        scores.append((text, float(sim)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


_PRIMARY_THRESHOLD = 0.6
_SECONDARY_THRESHOLD = 0.5

def _should_answer(scores):
    return (
        scores[0][1] >= _PRIMARY_THRESHOLD
        and scores[1][1] >= _SECONDARY_THRESHOLD
    )


def ai_system(query):
 
    if not isinstance(query, str) or not query.strip():
        return "INVALID_INPUT", "Please provide a valid query."

    query_embedding = _MODEL.encode([query])[0]

    scores = _retrieve(query_embedding)

    if _should_answer(scores):
        decision = "ANSWER_USING_MEMORY"
        response = "Based on what I know:\n"
        for text, _ in scores[:2]:
            response += f"- {text}\n"
    else:
        decision = "INSUFFICIENT_CONTEXT"
        response = "I don’t have enough information to answer that confidently."

    return decision, response


if __name__ == "__main__":
    test_queries = [
        "AI models work on numerical representations",
        "Does dopamine related to exercise",
        "What is an embedding?",
         "Politics influences public policy",
        
    ]

    for q in test_queries:
        decision, response = ai_system(q)
        print("\nQuery:", q)
        print("Decision:", decision)
        print("Response:\n", response)