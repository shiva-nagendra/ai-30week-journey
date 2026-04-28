from datasets import load_dataset

dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

new_data = []

for row in dataset["train"]:

    sample = {
        "messeges" : [
            {"role" : "system", "content" : "..."},
            {"role" : "user", "content" : row["instruction"]},
            {"role": "assistant", "content" : row["response"]}, 
        ]
    }

    new_data.append(sample)

instruction = row["instruction"].replace("{{orderNumber}}", "12345")




