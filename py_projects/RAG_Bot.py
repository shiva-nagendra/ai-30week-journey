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

#Embeddings
emb_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#VectorDB

vector_db = Chroma.from_documents(
    chunks,
    emb_model,
    persist_directory="py_projects/chroma_db"
)

print("\nVector DB created\n")

#Retriever
retriever = vector_db.as_retriever()

#Generator
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

llm = HuggingFacePipeline(pipeline=pipe)

#Query loop
while True:

    query = input("\nAsk question: (or type 'exit')")
    if query.lower() == "exit":
        break

    retrieved_docs = retriever._get_relevant_documents(query)

    context = "\n".join(
        doc.page_content for doc in retrieved_docs
    )



