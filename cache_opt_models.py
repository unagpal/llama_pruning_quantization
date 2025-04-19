import pandas as pd
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging

MODEL = "facebook/opt-1.3b"
MODEL_PATH = "./opt_models/"
def cache_model_nf4():
    model_dir = f"{MODEL_PATH}nf4"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True,
                                              trust_remote_code=True,
                                               use_auth_token=True)
    # nf4 quantization: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
    q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto",
        quantization_config = q_config, trust_remote_code=True, use_auth_token=True)
    os.makedirs(model_dir)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

def cache_model_int8():
    model_dir = f"{MODEL_PATH}int8"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True,
                                              trust_remote_code=True,
                                               use_auth_token=True)
    q_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q_config, device_map="auto",
                                             trust_remote_code=True, use_auth_token=True)
    os.makedirs(model_dir)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

def cache_model_f16():
    model_dir = f"{MODEL_PATH}f16"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True,
                                              trust_remote_code=True,
                                               use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                                 device_map="auto",
                                                                 trust_remote_code=True, 
                                                                 use_auth_token=True)
    os.makedirs(model_dir)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

cache_model_f16()
logging.info("F16 model is cached")
cache_model_nf4()
logging.info("NF4 model is cached")
cache_model_int8()
logging.info("INT8 model is cached")
