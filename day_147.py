#week 24 day 5
#Async RAG API

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio

