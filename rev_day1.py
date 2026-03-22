# REVISION DAY 1
# Semantic Search + Manual Transformer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


# -------- PART 1: SEMANTIC SEARCH -------- #

model_embed = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Artificial intelligence is transforming healthcare.",
    "Deep learning improves computer vision.",
    "Cooking requires heat and ingredients.",
    
]


doc_embeddings = model_embed.encode(documents)

query = "How do planes fly?"

query_embedding = model_embed.encode([query])

scores = cosine_similarity(query_embedding, doc_embeddings)[0]

top_index = np.argmax(scores)

print("\n[Semantic Search Result]")
print("Query:", query)
print("Best Match:", documents[top_index])
print("Score:", scores[top_index])

# -------- PART 2: TRANSFORMER (NO PIPELINE) -------- #

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "I love learning AI"

inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

pred = torch.argmax(probs).item()

labels = ["NEGATIVE", "POSITIVE"]

print("\n[Manual Transformer Result]")
print("Text:", text)
print("Prediction:", labels[pred])
print("Confidence:", probs[0][pred].item())