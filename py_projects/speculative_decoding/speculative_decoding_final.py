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

            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(current_tokens[0])

def speculative_decode(prompt, k=5, max_length=15, debug=False):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    current_tokens = inputs.input_ids

    with torch.no_grad():
        while current_tokens.shape[1] < max_length:
            draft_tokens = []
            draft_probs_list = []
            temp_tokens = current_tokens.clone()

            for _ in range(k):
                draft_output = draft_model(input_ids=temp_tokens)
                draft_logits = draft_output.logits[:, -1, :]
                draft_probs = F.softmax(draft_logits, dim=-1)

                next_draft_token = torch.multinomial(draft_probs, num_samples=1)

                draft_tokens.append(next_draft_token)
                draft_probs_list.append(draft_probs[0, next_draft_token.item()].item())

                temp_tokens = torch.cat([temp_tokens, next_draft_token], dim=1)

            draft_tokens_tensor = torch.cat(draft_tokens, dim=1)
            combined_tokens = torch.cat([current_tokens, draft_tokens_tensor], dim=1)
            target_output = target_model(input_ids=combined_tokens)

            N = current_tokens.shape[1]
            all_accepted = True
            eos_reached = False
            
            accepted_tokens = [] 

            for i in range(k):
                target_logits_i = target_output.logits[:, N - 1 + i, :]
                target_probs_i = F.softmax(target_logits_i, dim=-1)

                guessed_token = draft_tokens[i]
                p_draft = draft_probs_list[i]
                
                token_id = guessed_token.item()
                p_target = target_probs_i[0, token_id].item()

                r = torch.rand(1, device="cuda").item()
                ratio = min(1.0, (p_target / (p_draft + 1e-8)))

                if r < ratio:
                    accepted_tokens.append(guessed_token)
                    
                    if debug:
                        print(f"Iteration {i+1}: Accepted '{tokenizer.decode(guessed_token[0])}'")

                    if guessed_token.item() == tokenizer.eos_token_id:
                        eos_reached = True
                        break
                else:
                    new_target_token = torch.multinomial(target_probs_i, num_samples=1)
                    accepted_tokens.append(new_target_token)
                    
                    if debug:
                        print(f"Iteration {i+1}: Rejected '{tokenizer.decode(guessed_token[0])}', Sampled: '{tokenizer.decode(new_target_token[0])}'")
                    
                    all_accepted = False
                    
                    if new_target_token.item() == tokenizer.eos_token_id:
                        eos_reached = True
                    break

            if accepted_tokens:
                accepted_tensor = torch.cat(accepted_tokens, dim=1)
                current_tokens = torch.cat([current_tokens, accepted_tensor], dim=1)
            
            if all_accepted and not eos_reached:
                 next_logits = target_output.logits[:, N + k - 1, :]
                 next_probs = F.softmax(next_logits, dim=-1)
            
                 next_token = torch.multinomial(next_probs, num_samples=1)
                 current_tokens = torch.cat([current_tokens, next_token], dim=1)
                 
                 if debug:
                     print(f"Bonus token: '{tokenizer.decode(next_token[0])}'")
                 
                 if next_token.item() == tokenizer.eos_token_id:
                     eos_reached = True

            if eos_reached:
               if debug:
                   print("\n>> End of Sequence (EOS) reached. Stopping generation early.")
               break

    return tokenizer.decode(current_tokens[0])

if __name__ == "__main__":
    test_prompt = "The future of AI is"
    target_length = 20

    print("Running normal decoding: ")
    start_normal = time.time()
    normal_text = normal_decode(test_prompt, max_length=target_length)
    end_normal = time.time()

    print("\nNormal Decoding Result: \n", normal_text)
    print(f"Normal decoding time: {end_normal - start_normal:.4f} seconds")

    print("\nRunning speculative decoding: ")
    start_speculative = time.time()
    
    speculative_text = speculative_decode(test_prompt, k=5, max_length=target_length, debug=False)
    end_speculative = time.time()

    print("\nSpeculative Decoding Result: \n", speculative_text)
    print(f"Speculative decoding time: {end_speculative - start_speculative:.4f} seconds")

    speed_up = (end_normal - start_normal) / (end_speculative - start_speculative)
    print(f"\nSpeculative Decoding was {speed_up:.2f}x faster.")