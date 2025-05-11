from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
import torch

MODEL = "meta-llama/Llama-3.2-3B"
MODEL_PATH = "./llama_models/"
device = "cuda" if torch.cuda.is_available() else "cpu"

# GPTQ quantization using C4 dataset
def run_gptq (nbits: int):
    model_dir = f"{MODEL_PATH}bf16"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    gptq_config = GPTQConfig(
        bits=nbits,
        dataset="c4",
        block_size=128,
        tokenizer=tokenizer,
    )
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config = gptq_config, device_map="auto",
                                             local_files_only=True, trust_remote_code = True)
    model.to(device)
    model.eval()
    return model, tokenizer

# Script to compute and cache GPTQ quantized Llama models to local file system
def cache_gptq_models() -> None:
    model_8bit, tokenizer = run_gptq(8)
    model_8bit.save_pretrained(f"{MODEL_PATH}8bit", safe_serialization=True)
    model_4bit, tokenizer = run_gptq(4)
    model_4bit.save_pretrained(f"{MODEL_PATH}4bit", safe_serialization=True)

# Load GPTQ quantized Llama model and associated tokenizer
def load_llama_gptq (nbits: int):
    if nbits not in [4, 8]:
        raise ValueError("Only 4 and 8 bit quantized models are cached")

    model_dir = f"{MODEL_PATH}{nbits}bit"
    tok_dir = f"{MODEL_PATH}bf16"
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",
                                             local_files_only=True, trust_remote_code = True)
    model.to(device)
    model.eval()
    return model, tokenizer
