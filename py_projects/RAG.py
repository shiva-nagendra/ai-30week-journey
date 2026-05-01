#week 23 project
#RAG - Retrieval Augmented Generation

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data import data

# Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation",model="distilgpt2")

#Data
texts = [item["text"] for item in data]
doc_emb = embed_model.encode(texts)

#Main loop
while True:
    query = input("\nEnter your querry: (or exit)")
    if query == "exit".lower():
        break

    expanded_queries = [
        query,
        query + " in healthcare.",
        query + " medical applications."   
    ]
    
    all_scores = []

    for q in expanded_queries:
        q_emb = embed_model.encode([q])
        scores = cosine_similarity(q_emb, doc_emb)[0]
        all_scores.append(scores)

    final_scores = np.mean(all_scores, axis=0)

    top_indices = np.argsort(final_scores)[::-1][:3]

    ranked = []

    for idx in top_indices:
        text = texts[idx]
        score = final_scores[idx]

        #length penalty
        penalty = len(text.split()) * 0.01
        final_score = score - penalty

        ranked.append((text,final_score))

    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
   
    best_chunks = [r[0] for r in ranked[:3]]
    context = " ".join(best_chunks)

    print("\ncontext:\n", context)

    prompt = f"""
context : {context}
Question : {query}
Answer :
"""
    
    response = generator(prompt, max_length=120, num_return_sequences=1)
    answer = response[0]["generated_text"]

    print("\nAnswer:", answer)

    #Basic evaluation
    print("\nEvaluation:")

    for chunk in best_chunks:
        match = sum(word in chunk.lower() for word in query.lower())
        print(f"{chunk} --> relevance: {match}")






   








