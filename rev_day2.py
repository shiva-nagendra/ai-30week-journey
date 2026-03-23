# REVISION DAY 2
# Advanced Semantic Search + Manual Transformer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

