#week 20 day 3
#converting logits to prediction

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Using a standard model fine-tuned for SST-2 sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

#load
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "I love learning AI."

#Step 1 Tokenize
inputs = tokenizer(text, return_tensors = "pt")

#step 2 model forward
outputs = model(**inputs)

logits = outputs.logits

print("\nLogits:",logits)

#step 3: convert to probabilities

probs = torch.nn.functional.softmax(logits, dim=-1)

print("\nProbabilities:",probs)

#step 4: get prediction
pred_index = torch.argmax(probs).item()

labels = ["NEGATIVE", "POSITIVE"]

print("\nFinal prediction:", labels[pred_index])
print("Confidence:", probs[0][pred_index].item())