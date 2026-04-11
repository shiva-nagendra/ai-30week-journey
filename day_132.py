#week 22 day 2
#FAISS implementation

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")

