#week 25 day 3
#persistent chroma vector database

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os

#Load document
loader = TextLoader("medical_notes.txt")

documents = loader.load()

print(f"\nLoaded {len(documents)} doxument(s)\n")

#chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=30
)

chunks = splitter.split_documents(documents)

print(f"\nCreated {len(chunks)} chunks")

#Embedding model
emb_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#vector db path
db_path = "ai-30week-journey/db_path"

#create DB
if os.path.exists(db_path):
    print("Loading existing vector database..\n")

    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=emb_model
    )

else:
    print("Creating new vector databases..\n")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=emb_model,
        persist_directory=db_path
    )

print(f"stored chunks: {vector_db._collection.count()}")

while True:
    query = input("Enter your query (or 'exit')")
    if query.lower() == "exit":
        break

    