#week 20 day 1
#Understanding tokenizer + inputs

from transformers import AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "I love learning AI"

#tokenize
inputs = tokenizer(text)

print("\nOriginal text:",text)
print("\nTokenized output:",inputs)