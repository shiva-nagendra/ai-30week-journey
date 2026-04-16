#Speculative decoding

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch

#Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

#Load models
draft_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cuda")
target_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

draft_model.eval()
target_model.eval()

prompt = "AI is transforming "

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Intial token", prompt)

#intializing tokens with prompt
current_token = inputs.input_ids

with torch.no_grad():
    for i in range(5):
        #draft model output tokens on current sequence
        draft_output = draft_model(input_ids=current_token)
        draft_logits = draft_output.logits[:, -1, :]
        draft_prob = F.softmax(draft_logits, dim=-1)
        next_draft_token = torch.multinomial(draft_prob, num_samples=1)

        #Target model output tokens on current tokens
        target_output = target_model(input_ids=current_token)
        target_logits = target_output.logits[:, -1, :]
        target_probs = F.softmax(target_logits, dim=-1)

        #generate random r
        r = torch.rand(1, device="cuda")

        #draft and target probs tokens
        p_draft = draft_prob[0, next_draft_token.item()]
        p_target = target_probs[0, next_draft_token.item()]

        #logic
        if r < (p_target/p_draft):
            current_token = torch.cat([current_token, next_draft_token], dim=-1)
            print(f"Iteration{i+1}: Accepted: '{tokenizer.decode(next_draft_token[0])}' -> current position: '{tokenizer.decode(current_token[0])}'")

        else:
            new_token_target = torch.multinomial(target_probs,num_samples=1)
            current_token = torch.cat([current_token, new_token_target],dim=-1)
            print(f"Iteration{i+1}: Rejected'{tokenizer.decode(next_draft_token[0])}', token from target:'{tokenizer.decode(new_token_target[0])}', current position{tokenizer.decode(current_token[0])}")

            
        



