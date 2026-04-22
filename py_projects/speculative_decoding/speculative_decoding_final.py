#Final version of speculative decoding implementation
#Includes Ratio Stability, EOS Handling, and Bonus Token Generation
#Wrapped in function for easier reuse

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch
import time

tokenizer = AutoTokenizer.from_pretrained("gpt2")
target_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
draft_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cuda")

target_model.eval()
draft_model.eval()  

def normal_decode(prompt, max_length=15):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    current_tokens = inputs.input_ids

    with torch.no_grad():
        while current_tokens.shape[1] < max_length:

            target_output = target_model(input_ids=current_tokens)
            next_token_logits = target_output.logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
    