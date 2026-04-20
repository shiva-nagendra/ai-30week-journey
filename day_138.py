#week 23 day 2
#Retrieval + Generation

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

#Embedding model
emb_model = SentenceTransformer("all-MiniLM-L6-v2")

#Generation model
model_generator = pipeline("text-generation", model="distilgpt2")

#Data
documents = [
    "AI helps diagnose diseases",
    "Machine learning analyzes patient data",
    "Deep learning improves medical imaging",
    "AI is used in drug discovery",
    "Doctors use AI for treatment planning"
]

doc_emb = emb_model.encode(documents)

query = input("Enter your query: ")

query_emb = emb_model.encode([query])

#Retrieval
scores = cosine_similarity(query_emb, doc_emb)[0]
top_indices = np.argsort(scores)[::-1][:3]

context = " ".join([documents[idx] for idx in top_indices])

print("\nRetrieved context: ")
print(context)

#Generation

prompt = f"""
context: {context}

question: {query}

Answer:
"""

response = model_generator(prompt, max_length=100, num_return_sequences=1)

print("Generated responses: ")
print(response[0]["generated_text"])



