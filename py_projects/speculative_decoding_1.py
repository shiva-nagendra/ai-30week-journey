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

# Initialize tokens with the prompt
current_tokens = inputs.input_ids

print("Initial prompt:", tokenizer.decode(current_tokens[0]))

# Number of tokens to generate in total (excluding prompt)
num_total_tokens_to_generate = 20 # Example: Generate 20 new tokens
# Number of speculative tokens to draft at each step
k = 4 # You can adjust k, usually 4-7

generated_tokens_count = 0

with torch.no_grad():
  while generated_tokens_count < num_total_tokens_to_generate:
    # 1. Draft k tokens using the draft model
    speculative_tokens_list = []
    # Store full probability distributions from the draft model for each step
    draft_probs_for_speculative_tokens = []
    temp_current_tokens_for_draft = current_tokens.clone() # Use a temporary sequence for drafting

    for draft_step in range(k):
      draft_outputs = draft_model(input_ids=temp_current_tokens_for_draft)
      draft_logits = draft_outputs.logits[:, -1, :]
      draft_probs = F.softmax(draft_logits, dim=-1)
      next_token_draft_k = torch.multinomial(draft_probs, num_samples=1)

      speculative_tokens_list.append(next_token_draft_k)
      draft_probs_for_speculative_tokens.append(draft_probs) # Store the full distribution

      temp_current_tokens_for_draft = torch.cat([temp_current_tokens_for_draft, next_token_draft_k], dim=-1)
      # print(f"  Drafted token {draft_step+1}/{k}: '{tokenizer.decode(next_token_draft_k[0])}'")

    # 2. Evaluate the combined sequence (current_tokens + all drafted tokens) with the target model
    # The target model needs to see the current_tokens + ALL k drafted tokens at once
    full_sequence_for_target_eval = torch.cat([current_tokens] + speculative_tokens_list, dim=-1)
    target_outputs_eval = target_model(input_ids=full_sequence_for_target_eval)

    # Extract logits for the positions corresponding to the drafted tokens
    # The target model's logits at index `i` predict the token at `i+1`. So, for the first drafted token
    # (which starts after current_tokens), we look at logits starting from `len(current_tokens) - 1`.
    # target_logits_for_comparison will have shape (batch_size, k, vocab_size)
    target_logits_for_comparison = target_outputs_eval.logits[:, current_tokens.shape[1]-1 : current_tokens.shape[1]-1 + k, :]
    target_probs_for_comparison = F.softmax(target_logits_for_comparison, dim=-1)

    # 3. Acceptance/Rejection loop for drafted tokens
    num_accepted_in_this_step = 0
    break_early_sampling = False

    for m in range(k):
      if generated_tokens_count >= num_total_tokens_to_generate:
        break # Stop if we've generated enough tokens already

      t_m = speculative_tokens_list[m] # The m-th drafted token
      
      # Probability of t_m from draft model at step m of drafting (using the stored full distribution)
      p_draft_for_t_m = draft_probs_for_speculative_tokens[m][0, t_m.item()]

      # Probability of t_m from target model for this specific position 'm' (from the batch evaluation)
      p_target_for_t_m = target_probs_for_comparison[0, m, t_m.item()]

      r = torch.rand(1, device="cuda")

      # Apply the acceptance condition
      if r < (p_target_for_t_m / p_draft_for_t_m):
        # Accept t_m
        current_tokens = torch.cat([current_tokens, t_m], dim=-1)
        generated_tokens_count += 1
        num_accepted_in_this_step += 1
        print(f"  Accepted '{tokenizer.decode(t_m[0])}'. Current sequence: {tokenizer.decode(current_tokens[0])}")
      else:
        # Reject t_m, sample a new token from the target model's distribution for this position 'm'
        # For simplicity, if rejected, we sample directly from the target's distribution for that position.
        # A more rigorous approach involves sampling from (P_target(x) - P_draft(x))_+, normalized.
        adjusted_target_probs = target_probs_for_comparison[0, m, :]
        new_token_from_target = torch.multinomial(adjusted_target_probs, num_samples=1)
        current_tokens = torch.cat([current_tokens, new_token_from_target], dim=-1)
        generated_tokens_count += 1
        print(f"  Rejected '{tokenizer.decode(t_m[0])}'. Sampled '{tokenizer.decode(new_token_from_target[0])}' from target. Current sequence: {tokenizer.decode(current_tokens[0])}")
        break_early_sampling = True
        break # Break from the inner 'k' loop, as failure on any drafted token stops acceptance

    # If all k tokens were accepted, generate one more from the target to reduce bias (optional but common)
    if generated_tokens_count < num_total_tokens_to_generate and not break_early_sampling and num_accepted_in_this_step == k:
      # Sample one more from the target model based on the fully accepted sequence
      final_target_outputs_after_k = target_model(input_ids=current_tokens)
      final_target_logits_after_k = final_target_outputs_after_k.logits[:, -1, :]
      final_target_probs_after_k = F.softmax(final_target_logits_after_k, dim=-1)
      next_token_final_target = torch.multinomial(final_target_probs_after_k, num_samples=1)
      current_tokens = torch.cat([current_tokens, next_token_final_target], dim=-1)
      generated_tokens_count += 1
      print(f"  All {k} drafted tokens accepted. Sampled 1 additional token '{tokenizer.decode(next_token_final_target[0])}' from target. Current sequence: {tokenizer.decode(current_tokens[0])}")


print(f"\nFinal generated sequence (Total {generated_tokens_count} new tokens):")
print(tokenizer.decode(current_tokens[0]))