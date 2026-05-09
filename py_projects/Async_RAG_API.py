#week 24 project
#Production-style Async RAG API

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dotenv import load_dotenv
import os
import asyncio
import numpy

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



        


