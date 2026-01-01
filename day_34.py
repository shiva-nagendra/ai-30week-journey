# Models predict.
# Rules contextualize.
# Systems decide.
# #Text → Sentiment model → Confidence → Decision

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
    "inputs": "This product is bad.but I have mixed opinions on the service"
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

    THRESHOLD = 0.9 #confidence threshold

    if best["label"]== "POSITIVE" and best["score"] >= THRESHOLD:
            desion="APPROVED"
    elif best["label"]=="NEGATIVE" and best["score"] >= THRESHOLD:
         desion="REJECTED"
    else:
         desion="REVIEW"
    print("Decision:",desion)

    HIGH_CONF = 0.9
    LOW_CONF = 0.6

    score = best["score"]
    label = best["label"]

    if score >= HIGH_CONF:
         decision = f"AUTO_{label}"
    elif score >= LOW_CONF:
         decision = f"SOFT_{label}"
    else:
         decision = "UNSURE"
    print("Decision:",decision)
else:
    print("Error response:")
    print(response.text)

text = payload["inputs"].lower()

signals = {
    "sentiment": best["label"],
    "confidence": best["score"],
    "text_length": len(text),
    "has_contrast": "but" in text,
}

HIGH_CONF = 0.9
LOW_CONF = 0.6

if (
    signals["sentiment"] == "POSITIVE"
    and signals["confidence"] >= HIGH_CONF
    and signals["text_length"] > 20
    and not signals["has_contrast"]
):
    decision = "AUTO_APPROVED"

elif signals["confidence"] < LOW_CONF:
    decision = "UNSURE"

else:
    decision = "REVIEW"

print("Signals:", signals)
print("Final decision:", decision)