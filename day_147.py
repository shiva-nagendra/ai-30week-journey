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
    "Doctors use AI for treatment planning"

]

doc_emb = model.encode(documents)


