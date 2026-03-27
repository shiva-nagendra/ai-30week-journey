#week 20 day 3
#converting logits to prediction

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"

#load
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "I love learning AI."

