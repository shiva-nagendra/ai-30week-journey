#week 20 day 5
#deep dive into tokenizer

from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "I Love learning ai"

#step 1: tokenize (raw tokens)
tokens = tokenizer.tokenize(text)

print("Tokens:", tokens)

#convert tokens to ids
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\nToken IDs:", token_ids)

#step 3: Full encoding
encoding = tokenizer(text)

print("\nFull encoding", encoding)

#step 4: Decode back
decoded = tokenizer.decode(encoding["input_ids"])

print("\nDecoded back:", decoded)