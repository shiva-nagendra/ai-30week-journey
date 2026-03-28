#week 20 day 4
#Batch processing in transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

texts = [
    "I love AI",
    "This is Terrible.",
    "Machine learning is interesting."
]

#step 1: tokenize batch
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

print("Input IDs:\n", inputs["input_ids"])
print("\nAttention Mask:\n", inputs["attention_mask"])

#step 2: Model forward
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

print("\nLogits:\n",logits)

#step 3: softmax
probs = torch.nn.functional.softmax(logits, dim=-1)

#step 4: predictions
preds = torch.argmax(probs, dim=1)

labels = ["NEGATIVE", "POSITIVE"]

print("\nResults:")

for i, text in enumerate(texts):
    print(f"{text} - {labels[preds[i]]} ({probs[i][preds[i]].item():.3f})")