#week 20 day 2
#Understanding model outputs (logits)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

#Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "I love learning AI"

#step 1 tokenize
inputs = tokenizer(text, return_tensors ="pt")

print("Inputs\n", inputs)

#step 2 pass through model
outputs = model(**inputs)

print("\nRaw model output:\n", outputs)

#step 3 extract logits
logits = outputs.logits

print("\nLogits:\n",logits)