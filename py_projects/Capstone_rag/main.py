#week 25 capstone project
#FASTAPI RAG API

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

from dotenv import load_dotenv
import os

load_dotenv()

#API
app = FastAPI()

class QueryRequest(BaseModel):
    query:str

loader = PyPDFLoader(
    os.getenv("PDF_PATH")
    )

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

#Embedding Model
emb_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db_path = os.getenv("CHROMA_DB")

if os.path.exists(db_path):
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=emb_model
    )

else:
     vector_db = Chroma.from_documents(
         chunks,
         emb_model,
         persist_directory=db_path
     )

print("Vector_db created")

#retriever
retriever = vector_db.as_retriever()

pipe = pipeline(
    "text2text-generation",
    model=os.getenv("MODEL_NAME"),
    max_new_tokens=int(os.getenv("MAX_NEW_TOKENS"))
)

llm = HuggingFacePipeline(
    pipeline=pipe
)

#API
@app.post("/predict")

async def predict(req: QueryRequest):
    query = req.query

    retrieved_docs = retriever.invoke(query)

    context = "\n".join(
        doc.page_content for doc in retrieved_docs
    )

#prompt
    prompt = f"""

Answer only using the context below.

Context: 
{context}

Question:
{query}

Answer:
"""
    #Generate
    answer = llm.invoke(prompt)

    #json response
    return{
        "question": query,
        "answer": answer
    }

@app.get("/health")
def health():
    return {"status":"healthy"}






