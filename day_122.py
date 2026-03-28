#week 20 day 4
#Batch processing in transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

