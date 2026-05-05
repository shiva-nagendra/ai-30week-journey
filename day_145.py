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
model = SentenceTransformer("all-miniLM-L6-v2")
generator = pipeline("text-generation", model="distilgpt2")

documents = [

    "AI helps diagnose diseases",

    "Machine learning analyzes patient data",

    "Deep learning improves medical imaging",

    "AI is used in drug discovery",

    "Doctors use AI for treatment planning"

]

