#week 24 day 2
#API + RAG

from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

#INIT

app = FastAPI()

embed_model = SentenceTransformer("all-miniLM-L6-v2")
generator = pipeline("text-generation", model='distilgpt2')

documents = [

    "AI helps diagnose diseases",

    "Machine learning analyzes patient data",

    "Deep learning improves medical imaging",

    "AI is used in drug discovery",

    "Doctors use AI for treatment planning"

]

doc_emb = embed_model.encode(documents)

