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

    "Doctors use AI for treatment planning",

    "aeroplanes use bernouli's principle for flying"

]

doc_emb = model.encode(documents)

#Request format

class QueryRequest(BaseModel):
    query:str

#API Endpoint

@app.post("/ask")
def ask_question(req: QueryRequest):
    query = req.query

    query_emb = model.encode([query])

    scores = cosine_similarity(query_emb, doc_emb)[0]
    top_indices = np.argsort(scores)[::-1][:3]

    results = [documents[idx] for idx in top_indices]

    return {
        "query": query,
        "results": results
    }
    