from datasets import load_dataset
import json

dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

new_data = []

for row in dataset["train"]:
  modified_instruction = row["instruction"].replace("{{Order Number}}", "12345")

  # Check if the instruction indicates an order cancellation for the specific order number
  if ("cancel" in modified_instruction.lower() or "cancelling" in modified_instruction.lower()) and "12345" in modified_instruction:
      modified_response = f"""To cancel order 12345:
1. Log in to your account
2. Go to Orders
3. Select order 12345
4. Click Cancel

Contact support if needed."""
  else:
      # Apply existing placeholder replacements for other responses
      modified_response = row["response"].replace("{{Order Number}}", "12345")
      modified_response = modified_response.replace("{{Online Company Portal Info}}", "our customer portal")
      modified_response = modified_response.replace("{{Online Order Interaction}}", "Order History")
      modified_response = modified_response.replace("{{Customer Support Hours}}", "9 AM to 5 PM EST")
      modified_response = modified_response.replace("{{Customer Support Phone Number}}", "1-800-555-0123")
      modified_response = modified_response.replace("{{Website URL}}", "our website")

  sample = {
      "messages" : [
          {"role": "system", "content": "You are a business assistant helping users with orders and support."},
          {"role": "user", "content": modified_instruction},
          {"role": "assistant", "content": modified_response}
      ]
  }

  new_data.append(sample)


with open("train.jsonl", "w") as f:
    for item in new_data:
      f.write(json.dumps(item) + "\n")

print(new_data[:3])