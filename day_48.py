	# •	reasoning trace
	# •	decision explanation
	# •	structured logs

from sentence_transformers import SentenceTransformer
import math

_MEMORY = [
    "Regular exercise improves physical health",
    "Sports help maintain mental well-being",
    "AI models work on numerical representations",
    "Embeddings capture semantic meaning of text",
    "Policies are the rules for any system",
    "Politics influences public policy"
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

POLICY = {
    "high_confidence_single_evidence": 0.85,
    "primary_threshold": 0.6,
    "secondary_threshold": 0.5,
    "max_evidence_items": 2
}

def _should_answer(scores, policy):
    if scores[0][1] >= policy["high_confidence_single_evidence"]:
        return True, "HIGH_CONF_SINGLE_EVID"

    if (
        scores[0][1] >= policy["primary_threshold"]
        and scores[1][1] >= policy["secondary_threshold"]
    ):
        return True,"MIXED_EVID_MATCH"

    return False,"INSUFFICIENT_EVID"

def ai_system(query):
 
    if not isinstance(query, str) or not query.strip():
        return "INVALID_INPUT", "Please provide a valid query."

    query_embedding = _MODEL.encode([query])[0]

    scores = _retrieve(query_embedding)

    allowed, reason = _should_answer(scores, POLICY)
    if allowed:
        decision = "ANSWER_USING_MEM"
    else:
        decision = "INSUFFICIENT_CONTEXT"
        print("Decision reason:", reason)

    if _should_answer(scores, POLICY):
        decision = "ANSWER_USING_MEMORY"
        response = "Based on what I know:\n"
        for text, _ in scores[:2]:
            response += f"- {text}\n"
        print("\n Top retrieval scores:")
        for text, score in scores[:3]:
            print(f"{round(score,3)}-->{text}")
    else:
        decision = "INSUFFICIENT_CONTEXT"
        response = "I don’t have enough information to answer that confidently."

    return decision, response


if __name__ == "__main__":

    test_queries = [
        "AI models work on numerical representations",
        "",
        "67",
        "Does dopamine related to exercise",
        "What is an embedding?",
         "Politics influences public policy",
        
    ]

    for q in test_queries:
        decision, response = ai_system(q)
        print("\nQuery:", q)
        print("Decision:", decision)
        print("Response:\n", response)