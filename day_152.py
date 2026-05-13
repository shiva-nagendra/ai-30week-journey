#week 25 day 4
#Metadata filtering RAG

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

#create docs

documents = [
    Document(
        page_content="Ai helps docs diagnose diseases.",
        metadata={"topic":"healthcare"}
    ),

    Document(
        page_content="Machine learning predicts stock prices.",
        metadata={"topic":"finance"}
    ),

    Document(
        page_content="AI assists lawyers in legal research",
        metadata={"topic": "legal"}
    ),

    Document(
        page_content="Deep learning improves medical imaging",
        metadata={"topic": "healthcare"}
    )    
]

#embeddings
emb_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#vector DB
vector_db = Chroma.from_documents(
    documents,
    emb_model
)

#Retriever
retriever = vector_db.as_retriever()

#Query
query = input("Ask question: ")

#filtered search
results = vector_db.similarity_search(
    query,
    k=3,
    filter={"topic":"healthcare"}
)

#output
print("\nFiltered results:\n")

for idx, doc in enumerate(results):
    print(f"Result {idx+1}:\n")

    print(doc.page_content)

    print(f"\nMetadata: {doc.metadata}")