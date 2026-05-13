#week 25 day 5
#Hybrid Search (Keyword + Semantic)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#docs
documents = [

    "GPT-4 Turbo supports large context windows",

    "Machine learning improves healthcare systems",

    "Python is popular for AI engineering",

    "GPT-4 pricing depends on token usage",

    "Hospitals use AI for medical imaging"

]

