#week 23 day 4
#Chunkning ranking for RAG

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np      
from transformers import pipeline

