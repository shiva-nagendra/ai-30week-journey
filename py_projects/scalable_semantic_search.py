#week 22 project
#Scalable semantic search using FAISS

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from data_week22 import data

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

