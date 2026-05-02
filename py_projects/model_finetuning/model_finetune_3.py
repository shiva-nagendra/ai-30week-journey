from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
import os
from google.colab import userdata
import torch


# Get HF_TOKEN from Colab Secrets and set as environment variable
HF_TOKEN = userdata.get('HF_TOKEN')
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
    print("Hugging Face token loaded from secrets.")
else:
    print("Warning: HF_TOKEN not found in Colab secrets. Model downloads might be rate-limited.")

#models
model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set pad_token for the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Set a chat template for the tokenizer
tokenizer.chat_template = "{% for message in messages %}{{'<|' + message['role'] + '|>: ' + message['content'] + '\n'}}{% endfor %}"

# Load model config and set pad_token_id
config = AutoConfig.from_pretrained(model_name)
config.pad_token_id = tokenizer.pad_token_id


# Define BitsAndBytesConfig for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
    
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    config=config # Pass the modified config
)

#Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Enable gradient checkpointing and disable cache for memory optimization
model.gradient_checkpointing_enable()
model.config.use_cache = False

#load data
dataset = load_dataset("json", data_files="train.jsonl")

#Trainer setup
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    output_dir="./results",
    fp16=False,
    bf16=False
)

def format_chat(example):
    text = ""
    for msg in example["messages"]:
        text += f"<|{msg['role']}|>: {msg['content']}\n"
    return text


#Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"].select(range(2000)),
    args=training_args,
    processing_class=tokenizer,
    formatting_func=format_chat
)

trainer.train()
