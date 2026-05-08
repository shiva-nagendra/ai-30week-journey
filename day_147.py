#week 24 day 5
#Async RAG API

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)


documents = [

    "AI helps diagnose diseases",
    "Machine learning analyzes patient data",
    "Deep learning improves medical imaging",
    "AI is used in drug discovery",
    "Doctors use AI for treatment planning",
    "AI can be used to guess the patterns of disease evolution and virus spread."

]

doc_emb = model.encode(documents)

cache={}

#Request model

class QueryRequest(BaseModel):
    query:str

async def generate_stream(response_text):

    words = response_text.split()

    for word in words:
        yield word + " "
        await asyncio.sleep(0.03)

#API
@app.post("/ask-stream")
async def ask_stream(req: QueryRequest):

    query = req.query.strip().lower()

    if query in cache:
        return StreamingResponse(
            generate_stream(cache[query]),
            media_type="text/plain"
        )
    
    #Retrieval

    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, doc_emb)[0]
    top_indices = np.argsort(scores)[::-1][:3]

    context = " ".join(documents[idx] for idx in top_indices)


#prompt


    prompt = f"""
Answer the question clearly using the context.

Context:
{context}

Question:
{query}

Answer:
"""
    response = generator(
        prompt,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.3
    )[0]["generated_text"]

    return StreamingResponse(
        generate_stream(response),
        media_type="text/plain"
    )
    
