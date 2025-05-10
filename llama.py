import numpy as np
from transformers import BitsAndBytesConfig, LlamaForCausalLM, AutoTokenizer
import torch
from torch import nn
from torch.nn.utils import prune
import time
import logging

MODEL = "meta-llama/Llama-3.2-3B"
MODEL_PATH = "./llama_models/"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_nf4_bitsandbytes():
    model_dir = f"{MODEL_PATH}nf4"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    # nf4 quantization: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
    q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_use_double_quant=True)
    model = LlamaForCausalLM.from_pretrained(model_dir, device_map="auto",
        quantization_config = q_config, local_files_only=True)
    return model, tokenizer

def load_model_int8_bitsandbytes():
    model_dir = f"{MODEL_PATH}int8"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    q_config = BitsAndBytesConfig(load_in_8bit=True)
    model = LlamaForCausalLM.from_pretrained(model_dir, quantization_config=q_config, device_map="auto",
                                             local_files_only=True)
    return model, tokenizer

def load_model_bf16():
    model_dir = f"{MODEL_PATH}bf16"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto",
                                             local_files_only=True)
    model.to(device)
    model.eval()
    return model, tokenizer

# Standard bf16 model load
def load_model_bf16_bmk ():
    model_dir = f"{MODEL_PATH}bf16"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto",
                                             local_files_only=True)
    model.to(device)
    model.eval()
    return model, tokenizer

