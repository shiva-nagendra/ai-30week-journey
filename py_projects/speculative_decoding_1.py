#Speculative decoding research implementation

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

#Load models
target_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
draft_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cuda")

target_model.eval()  
draft_model.eval()

#Task
prompt = "AI is transforming"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
  outputs = target_model(**inputs)

logits = outputs.logits[:,-1,:]
print(logits.shape)

probs = torch.nn.functional.softmax(logits, dim=-1)

next_token_sample = torch.multinomial(probs, num_samples=4)

print("\nNext token:")

print(tokenizer.decode(next_token_sample[0]))






