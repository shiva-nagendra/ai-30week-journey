#week 24 project
#Framework based RAG bot
#LangChain + ChromaDB

from langchain_community.document_loaders import PyPDFLoader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline

from transformers import pipeline

#Load pdf

loader = PyPDFLoader("py_projects/AI Project Mentorship_ Research & Fine-Tuning - Google Docs.pdf")

docs = loader.load()

print(f"Loaded {len(docs)} pages\n")

#Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks\n")



