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