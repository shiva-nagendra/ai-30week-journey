
# Hugging Face inference (current API) and post decision


import requests
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

HF_TOKEN = os.getenv("HF_TOKEN") #HF_TOKE = HF_TOKEN
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set")


API_URL = (
    "https://router.huggingface.co/hf-inference/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

payload = {
    "inputs": "I love learning AI step by step"
}

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

response = requests.post(
    API_URL,
    headers=headers,
    json=payload
)

print("Status:", response.status_code)

if response.status_code == 200:
    result = response.json()[0] 

    best = max(result, key=lambda x: x["score"])

    print("Sentiment:", best["label"])
    print("Confidence:", round(best["score"], 3))
else:
    print("Error response:")
    print(response.text)

THRESHOLD = 0.9 #confidence threshold

if best["label"]== "POSITIVE" and best["score"] >= THRESHOLD:
        desion="APPROVED"
elif best["label"]=="NEGATIVE" and best["score"] >= THRESHOLD:
     desion="REJECTED"
else:
     desion="REVIEW"
print("Decision:",desion)