#Speculative Decoding implementation
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# 1. Load Tokenizer and Models
tokenizer = AutoTokenizer.from_pretrained("gpt2")
draft_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cuda")
target_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

draft_model.eval()
target_model.eval()

prompt = "AI is transforming "
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print(f"Initial prompt: '{prompt}'\n")

current_tokens = inputs.input_ids
k = 5

with torch.no_grad():
    # PHASE 1: Draft model generates K tokens
    draft_tokens = []
    draft_probs_list = [] # We need to remember the math for each guess!
    temp_tokens = current_tokens.clone()

    for _ in range(k):
        # Generate guess
        draft_output = draft_model(input_ids=temp_tokens)
        draft_logits = draft_output.logits[:, -1, :]
        draft_prob = F.softmax(draft_logits, dim=-1)
        
        next_draft_token = torch.multinomial(draft_prob, num_samples=1)
        
        # Save the token AND the probability the draft model assigned to it
        draft_tokens.append(next_draft_token)
        draft_probs_list.append(draft_prob[0, next_draft_token.item()])
        
        # Append for the next draft loop
        temp_tokens = torch.cat([temp_tokens, next_draft_token], dim=1)

    # Convert our list of k tokens into a single tensor
    draft_tokens_tensor = torch.cat(draft_tokens, dim=1)

    # PHASE 2: Target model verifies ALL at once

    # Combine original prompt + all k draft guesses
    combined_tokens = torch.cat([current_tokens, draft_tokens_tensor], dim=1)

    # ONE single forward pass! (This is where the speedup happens)
    target_output = target_model(input_ids=combined_tokens)

    # Calculate where the original prompt ends so we know where to start checking
    N = current_tokens.shape[1] 

    # PHASE 3: Accept / Reject Loop
    all_accepted = True

    for i in range(k):
        # Get the target model's math for this specific position
        # Position is N - 1 (last prompt token) + i (current draft token)
        target_logits_at_i = target_output.logits[:, N - 1 + i, :]
        target_probs_at_i = F.softmax(target_logits_at_i, dim=-1)

        # Get the specific token the draft model guessed here
        guessed_token = draft_tokens[i]

        # Probabilities for the accept/reject math
        p_draft = draft_probs_list[i]
        p_target = target_probs_at_i[0, guessed_token.item()]

        # The Speculative Decoding Accept/Reject formula
        r = torch.rand(1, device="cuda").item()

        if r < (p_target / p_draft):
            # ACCEPTED
            current_tokens = torch.cat([current_tokens, guessed_token], dim=1)
            print(f"Iteration {i+1}: Accepted '{tokenizer.decode(guessed_token[0])}'")
        else:
            # REJECTED
            # Sample a new, correct token directly from the Target model's math
            correct_token = torch.multinomial(target_probs_at_i, num_samples=1)
            current_tokens = torch.cat([current_tokens, correct_token], dim=1)
            
            print(f"Iteration {i+1}: REJECTED '{tokenizer.decode(guessed_token[0])}'. Replaced with '{tokenizer.decode(correct_token[0])}'")
            
            # Stop the loop
            all_accepted = False
            break

    print(f"\nFinal Sequence: {tokenizer.decode(current_tokens[0])}")