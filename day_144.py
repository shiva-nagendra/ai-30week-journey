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

#Request format
class QueryRequest(BaseModel):
    query:str

@app.post("/ask")
def ask_question(req: QueryRequest):
    query = req.query

    #Retrieval
    query_emb = embed_model.encode([query])
    scores = cosine_similarity(query_emb, doc_emb)[0]
    top_indices = np.argsort(scores)[::-1][:3]

    context = " ".join([documents[idx] for idx in top_indices])
    
    #generation
    prompt = f"""

    

