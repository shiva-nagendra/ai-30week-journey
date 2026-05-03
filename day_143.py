#week 24 day 1
#FastAPI

from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#INIT

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [

    "AI helps diagnose diseases",

    "Machine learning analyzes patient data",

    "Deep learning improves medical imaging",

    "AI is used in drug discovery",

    "Doctors use AI for treatment planning"

]