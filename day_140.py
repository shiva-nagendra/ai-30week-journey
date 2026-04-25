#week 23 day 4
#Chunkning ranking for RAG

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np      
from transformers import pipeline

# Models

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="distilgpt2")

# Data

documents = [

    "AI helps diagnose diseases",

    "Machine learning analyzes patient data",

    "Deep learning improves medical imaging",

    "AI is used in drug discovery",

    "Doctors use AI for treatment planning"

]

doc_embeddings = embed_model.encode(documents)

# -------- QUERY -------- #

query = input("Enter your question: ")
query_emb = embed_model.encode([query])

# -------- STEP 1: RETRIEVE MORE -------- #

k = 5

scores = cosine_similarity(query_emb, doc_embeddings)[0]

top_indices = np.argsort(scores)[::-1][:k]

# -------- STEP 2: RERANK -------- #

ranked_chunks = []

for idx in top_indices:

    text = documents[idx]

    # simple reranking: length + similarity

    score = cosine_similarity(query_emb, [doc_embeddings[idx]])[0][0]

    # bonus: prefer shorter, more direct chunks

    length_penalty = len(text.split()) * 0.01

    final_score = score - length_penalty

    ranked_chunks.append((text, final_score))

    # Sort again

ranked_chunks = sorted(ranked_chunks, key=lambda x: x[1], reverse=True)

# -------- STEP 3: SELECT BEST -------- #

best_chunks = [chunk[0] for chunk in ranked_chunks[:3]]

context = " ".join(best_chunks)

print("\nBetter Context:\n")

print(context)

