import pandas as pd
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import logging
MODEL = "facebook/opt-1.3b"
MODEL_PATH = "./opt_models/"

def load_model_f16():
    model_dir = f"{MODEL_PATH}f16"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def race_entry_to_prompt (row: pd.Series) -> str:
    if len(row["options"]) != 4:
        raise ValueError("Invalid race question format: 4 answer choices expected")

    return f"""Article: {row["article"]}
Question: {row.question}
Select the best choice from the following options:
A) {row["options"][0]}
B) {row["options"][1]}
C) {row["options"][2]}
D) {row["options"][3]}
    """

def load_data ():
    df = pd.read_parquet("race_high_test")
    df["prompt"] = df.apply(race_entry_to_prompt, axis=1)
    return df

model, tokenizer = load_model_f16()
print('LOADED MODEL!')
model.to("cuda")
model.eval()
