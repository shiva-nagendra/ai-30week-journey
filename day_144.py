#week 24 day 2
#API + RAG

from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

