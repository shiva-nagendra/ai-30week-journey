#week 19 day 4
#Sentiment analysis without pipeline

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

#input text
text = "I love training AI"

#tokenize text
inputs = tokenizer(text, return_tensors="pt")

#Run model
outputs = model(**inputs)

#convert logits to probabilities
probs = torch.nn.functional.softmax(outputs.logits, dim=1)

#get prediction
predicted_class = torch.argmax(probs).item()

labels = ["NEGATIVE", "POSITIVE"]

print("Text:", text)
print("prediction:", labels[predicted_class])
print("confidence:", probs[0][predicted_class].item())