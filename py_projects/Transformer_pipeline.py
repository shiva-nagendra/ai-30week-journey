#week 20 project
#Full Transformers pipeline

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model
model_name = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    ignore_mismatched_sizes = True)

