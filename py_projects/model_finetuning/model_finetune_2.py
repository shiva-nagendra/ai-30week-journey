#iteration 2 of model finetuning using QLoRA

from transformers import autotokenizer, AutoModelForCausalLM, training_arguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

#models 
model_name = "microsoft/phi-2"

tokenizer = autotokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

#Apply LoRA

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CASUAL_LM"
)

model = get_peft_model(model, lora_config)

#load data
dataset = load_dataset("json", data_files="train.jsonl")



