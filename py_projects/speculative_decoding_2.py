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
           draft_probs_list.append(draft_probs[0, next_draft_token.item()])

           temp_tokens = torch.cat([temp_tokens, next_draft_token], dim=1)

        draft_tokens_tensor = torch.cat(draft_tokens, dim=1)

        combined_tokens = torch.cat([current_tokens, draft_tokens_tensor], dim=1)

        target_output = target_model(input_ids=combined_tokens)

        N = current_tokens.shape[1]

        all_accepted = True

        for i in range(k):
            target_logits_i = target_output.logits[:,N-1+i,:]
            target_probs_i = F.softmax(target_logits_i, dim=-1)

            guessed_token = draft_tokens[i]

            p_draft = draft_probs_list[i]
            p_target = target_probs_i[0, guessed_token.item()]

            r = torch.rand(1, device="cuda").item()

            if r < (p_target/p_draft):
                current_tokens = torch.cat([current_tokens, guessed_token],dim=1)
                print(f"Iteration{i+1}: Accepted '{tokenizer.decode(guessed_token[0])}'")

            else:
                new_target_tokens = torch.multinomial(target_probs_i, num_samples=1)
                current_tokens = torch.cat([current_tokens, new_target_tokens], dim=1)
                print(f"Iteration{i+1}: Rejected '{tokenizer.decode(guessed_token[0])}', Sampled: {tokenizer.decode(new_target_tokens[0])}")

                all_accepted=False
                break
                
        if all_accepted:
            next_logits = target_output.logits[:, N + k - 1, :]
            next_probs = F.softmax(next_logits, dim=-1)
            
            next_token = torch.multinomial(next_probs, num_samples=1)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            print("Bonus token:", tokenizer.decode(next_token[0]))

print("Final generation: ", tokenizer.decode(current_tokens[0]))