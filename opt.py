import pandas as pd
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import logging
MODEL = "facebook/opt-1.3b"
MODEL_PATH = "./opt_models/"

# Load float16 OPT model from local file system
def load_model_f16():
    model_dir = f"{MODEL_PATH}f16"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
    model.to("cuda")
    model.eval()
    return model, tokenizer
