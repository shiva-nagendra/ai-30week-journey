#week 20 project
#Full Transformers pipeline

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model
model_name = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    ignore_mismatched_sizes = True)

# Function

def predict_sentiment_text(text):

    inputs = tokenizer(text, return_tensors = "pt")

    print("\nInput Ids:", inputs["input_ids"])
    print("Attention mask:\n", inputs["attention_mask"])


    with torch.no_grad():
        outputs = model(**inputs)
