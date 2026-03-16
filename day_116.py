#week 19 day 4
#Sentiment analysis without pipeline

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

