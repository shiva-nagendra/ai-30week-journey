from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

# Load the tokenizer first
tokenizer = AutoTokenizer.from_pretrained("phi2-business-assistant")
tokenizer.pad_token = tokenizer.eos_token

# Load model config and set pad_token_id
config = AutoConfig.from_pretrained("microsoft/phi-2")
config.pad_token_id = tokenizer.pad_token_id

# Define BitsAndBytesConfig for 4-bit quantization (same as training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config, # Apply 4-bit quantization
    config=config # Pass the modified config
)

model = PeftModel.from_pretrained(base_model, "phi2-business-assistant")
model.to("cuda") # Move the entire model to CUDA

model.eval()

def generate_response(prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id # Explicitly pass pad_token_id for generation
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "<|system|>: You are a business assistant.\n<|user|>: How do I cancel order 67890?\n<|assistant|>:"
print(generate_response(prompt))