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



