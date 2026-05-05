#week 24 day 3
#streaming RAG API

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import time

app = FastAPI()

#model
model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-small")

documents = [

    "AI helps diagnose diseases",

    "Machine learning analyzes patient data",

    "Deep learning improves medical imaging",

    "AI is used in drug discovery",

    "Doctors use AI for treatment planning"

]

doc_emb = model.encode(documents)

class QueryRequest(BaseModel):
    query: str

def generate_stream(prompt):
    response = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False
    )[0]["generated_text"]

    words = response.split()

    for word in words:
        yield word + " "
        time.sleep(0.05)

#API
@app.post("/ask-stream")
def ask_stream(req: QueryRequest):
    query = req.query

    #Retrieval
    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, doc_emb)[0]
    top_indices = np.argsort(scores)[::-1][:3]

    context = " ".join([documents[idx] for idx in top_indices])

    #prompt
    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    return StreamingResponse(generate_stream(prompt), media_type="text/plain")
