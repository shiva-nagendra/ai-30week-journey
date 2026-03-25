# REVISION DAY 2
# Advanced Semantic Search + Manual Transformer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# -------- PART 1: SEMANTIC SEARCH (TOP-K) -------- #

model_embed = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Artificial intelligence is transforming healthcare.",
    "Machine learning helps analyze data.",
    "Deep learning improves image recognition.",
    "Airplanes fly using lift and aerodynamics.",
    "Cooking requires proper ingredients and heat.",
    "Programming builds software systems.",
    "Apple has it's own powerful silicon for macbooks.",
    "Apple launched Macbook neo for entry level line up.",
    "Macbook m5 model released in 2026 and it is very powerful."
]

doc_embeddings = model_embed.encode(documents)

#user query
query = input("Enter your query: ")

query_embedding = model_embed.encode([query])

scores = cosine_similarity(query_embedding, doc_embeddings)[0]

# top 3 matches
top_indices = np.argsort(scores)[::-1][:3]

print("\n[Top Matches]\n")

for idx in top_indices:
    print(f"{documents[idx]} (score: {scores[idx]:.3f})")

# -------- PART 2: TRANSFORMER MULTI-TEST -------- #

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

texts = [
    "I love this product",
    "This is terrible",
    "I am not sure about this"
]

labels = ["NEGATIVE", "POSITIVE"]

print("\n[Sentiment Results]\n")

for text in texts:

    inputs = tokenizer(text, return_tensors="pt")

    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    pred = torch.argmax(probs).item()

    print(f"{text} → {labels[pred]} ({probs[0][pred].item():.3f})")