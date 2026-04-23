#week 23 day 3
#Query improvement + RAG

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

#Embedding model
emb_model = SentenceTransformer("all-MiniLM-L6-v2")
#Generation model
generator = pipeline("text-generation", model="distilgpt2")


#data
documents = [

    "AI helps diagnose diseases",

    "Machine learning analyzes patient data",

    "Deep learning improves medical imaging",

    "AI is used in drug discovery",

    "Doctors use AI for treatment planning"

]

doc_emb = emb_model.encode(documents)

query = input("Enter your query: ") 

expanded_query = [
    query,
    query + " in healthcare",
    query + " for diagnosis",
    "machine learning" + query,
]

#Retrieval

all_scores = []

for q in expanded_query:
    q_emb = emb_model.encode([q])
    scores = cosine_similarity(q_emb, doc_emb)[0]
    all_scores.append(scores)

#combine scores
final_scores = np.mean(all_scores, axis=0)
top_indices = np.argsort(final_scores)[::-1][:3]
context = " ".join([documents[idx] for idx in top_indices])
print("\nRetrieved context: ")
print(context)

#Generation
prompt = f"""
context: {context},
question: {query},
Answer:
"""
