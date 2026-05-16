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

import os

app = FastAPI()

class QueryRequest(BaseModel):
    query:str

loader = PyPDFLoader(
    "py_projects/Capstone_rag/docs/ai_pdf.pdf"
    )

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

emb_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db_path = "py_projects/Capstone_rag/chroma_db"

vector_db = Chroma(
    persist_directory=db_path,
    embedding_function=emb_model
)








