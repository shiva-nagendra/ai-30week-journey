#week 24 project
#Production-style Async RAG API

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dotenv import load_dotenv
import os
import asyncio
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from data import documents

#Load env
load_dotenv()

APP_MODE = os.getenv("APP_MODE")
MODEL_NAME = os.getenv("MODEL_NAME")

print(f"Running in: {APP_MODE}")

#FastAPI
app = FastAPI()

#Load models

model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline(
    "text2text-generation",
    model=MODEL_NAME
)

#doc embeddings
doc_emb = model.encode(documents)

#cache
cache={}

#Request model
class QueryRequest(BaseModel):
    query:str

#streaming
async def generate_stream(response_text):

    words = response_text.split()

    for word in words:
        yield word + " "
        await asyncio.sleep(0.03)

#API 
@app.post("/ask")
async def ask(req: QueryRequest):

    query = req.query.strip()

    #cache check
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
You are a helpful AI assistant that answers using the context

Context:
{context}

Question:
{query}

Answer in 3 sentences
"""
    #Generation
    response = generator(
        prompt,
        max_new_tokens=80,
        repetition_penalty=1.4,
        do_sample=True,
        temperature=0.5
    )[0]["generated_text"]

    #cache store
    cache[query] = response

    #stream response
    return StreamingResponse(
        generate_stream(response),
        media_type="text/plain"
    )


    
    

        


