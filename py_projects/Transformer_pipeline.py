#week 20 project
#Full Transformers pipeline

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

