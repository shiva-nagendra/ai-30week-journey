# day_31.py
# Hugging Face inference (current API)

# day_31.py
# Hugging Face inference (router API – correct & safe)

# day_31.py
# Day 31: First real AI model inference (Hugging Face)

import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(dotenv_path=".env")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set")

# 1. Model endpoint (router – current HF API)
API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# 2. Input to the model
payload = {
    "inputs": "I love learning AI step by step"
}

# 3. Headers (identity + format)
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# 4. API call → THIS creates `response`
response = requests.post(
    API_URL,
    headers=headers,
    json=payload
)

print("Status:", response.status_code)

# 5. Safely handle the response
if response.status_code == 200:
    result = response.json()[0]   # first (and only) input’s predictions

    best = max(result, key=lambda x: x["score"])

    print("Sentiment:", best["label"])
    print("Confidence:", round(best["score"], 3))
else:
    print("Error response:")
    print(response.text)
