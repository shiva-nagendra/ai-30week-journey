#Speculative Decoding implementation
#Added Ratio Stability, EOS Handling, and Bonus Token Generation

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
target_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
draft_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cuda")

target_model.eval()
draft_model.eval()

prompt = "AI is transforming.."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Intial prompt: ", prompt)

current_tokens = inputs.input_ids

k = 5
max_length = 15

with torch.no_grad():
    
    while current_tokens.shape[1] < max_length:
        draft_tokens = []
        draft_probs_list = []
        temp_tokens = current_tokens.clone()

        for _ in range(k):
           draft_output = draft_model(input_ids=temp_tokens)
           draft_logits = draft_output.logits[:,-1,:]
           draft_probs = F.softmax(draft_logits, dim=-1)

           next_draft_token = torch.multinomial(draft_probs, num_samples=1)

           draft_tokens.append(next_draft_token)
           
           # MODIFICATION 2: Added .item() here
           draft_probs_list.append(draft_probs[0, next_draft_token.item()].item())

           temp_tokens = torch.cat([temp_tokens, next_draft_token], dim=1)

        draft_tokens_tensor = torch.cat(draft_tokens, dim=1)
        combined_tokens = torch.cat([current_tokens, draft_tokens_tensor], dim=1)
        target_output = target_model(input_ids=combined_tokens)

        N = current_tokens.shape[1]
        all_accepted = True
        eos_reached = False # NEW: Flag to track if the AI is finished speaking

        for i in range(k):
            target_logits_i = target_output.logits[:, N-1+i, :]
            target_probs_i = F.softmax(target_logits_i, dim=-1)

            guessed_token = draft_tokens[i]

            p_draft = draft_probs_list[i]
            
            # Extract as a pure Python float for clean math
            p_target = target_probs_i[0, guessed_token.item()].item() 

            r = torch.rand(1, device="cuda").item()

            # MODIFICATION 1: Ratio Stability
            ratio = min(1.0, (p_target / (p_draft + 1e-8)))

            if r < ratio:
                current_tokens = torch.cat([current_tokens, guessed_token],dim=1)
                print(f"Iteration {i+1}: Accepted '{tokenizer.decode(guessed_token[0])}'")
                
                # MODIFICATION 3: EOS Handling (Accept)
                if guessed_token.item() == tokenizer.eos_token_id:
                    eos_reached = True
                    break

            else:
                new_target_tokens = torch.multinomial(target_probs_i, num_samples=1)
                current_tokens = torch.cat([current_tokens, new_target_tokens], dim=1)
                print(f"Iteration {i+1}: Rejected '{tokenizer.decode(guessed_token[0])}', Sampled: {tokenizer.decode(new_target_tokens[0])}")

                all_accepted = False
                
                # MODIFICATION 3: EOS Handling (Reject)
                if new_target_tokens.item() == tokenizer.eos_token_id:
                    eos_reached = True
                break
                
        if all_accepted and not eos_reached:
            next_logits = target_output.logits[:, N + k - 1, :]
            next_probs = F.softmax(next_logits, dim=-1)
            
            next_token = torch.multinomial(next_probs, num_samples=1)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            print("Bonus token:", tokenizer.decode(next_token[0]))
            
            # MODIFICATION 3: EOS Handling (Bonus)
            if next_token.item() == tokenizer.eos_token_id:
                eos_reached = True

        # If any of the steps hit the End of Sequence token, kill the while loop!
        if eos_reached:
            print("\n>> End of Sequence (EOS) reached. Stopping generation early.")
            break

print("\nFinal generation: ", tokenizer.decode(current_tokens[0]))