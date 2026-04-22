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

            eos_token_id = tokenizer.eos_token_id
            if next_token.item() == eos_token_id:
                break
    return tokenizer.decode(current_tokens[0])

def speculative_decode(prompt, k=5, max_length=15):
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

            for i in range(k):
                target_logits_i = target_output.logits[:, N-1+i, :]
                target_probs_i = F.softmax(target_logits_i, dim=-1)

                guessed_token = draft_tokens[i]
                p_draft = draft_probs_list[i]
                p_target = target_probs_i[0, guessed_token.item()].item()

                if p_target < p_draft:
                    all_accepted = False
                    break

                if guessed_token.item() == tokenizer.eos_token_id:
                    eos_reached = True
                    break

            if all_accepted:
                current_tokens = torch.cat([current_tokens, draft_tokens_tensor[:, :i+1]], dim=1)
            else:
                next_token_logits = target_output.logits[:, N-1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)

            if eos_reached:
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
    speculative_text = speculative_decode(test_prompt, k=5, max_length=target_length)
    end_speculative = time.time()

    print("\nSpeculative Decoding Result: \n", speculative_text)
    print(f"Speculative decoding time: {end_speculative - start_speculative:.4f} seconds")

    #Calculate the speedup
    speed_up = (end_normal - start_normal) / (end_speculative - start_speculative)
    print(f" Speculative Decoding was {speed_up:.2f}x faster.")
  
