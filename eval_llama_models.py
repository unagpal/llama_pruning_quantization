import pandas as pd
import re
from transformers import BitsAndBytesConfig, LlamaForCausalLM, AutoTokenizer
import torch
from torch import nn
from torch.nn.utils import prune
import time
import logging
import matplotlib.pyplot as plt
MODEL = "meta-llama/Llama-3.2-3B"
MODEL_PATH = "./llama_models/"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_nf4():
    model_dir = f"{MODEL_PATH}nf4"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    # nf4 quantization: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
    q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_use_double_quant=True)
    model = LlamaForCausalLM.from_pretrained(model_dir, device_map="auto",
        quantization_config = q_config, local_files_only=True)
    return model, tokenizer

def load_model_int8():
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

def load_bf16_model_structured_pruning(compression_ratio: float):
    model, tokenizer = load_model_bf16()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            prune.ln_structured(
                layer,
                name="weight",
                amount=compression_ratio, 
                n=1,
                dim=0
            )
            prune.remove(layer, "weight")

    model.to(device)
    model.eval()
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
Answer: """

def load_data ():
    df = pd.read_parquet("race_high_test")
    df["prompt"] = df.apply(race_entry_to_prompt, axis=1)
    return df

def run_bf16_exp():
    THROUGHPUT_TOKENS = 100
    N_SAMPLES = 100
    model, tokenizer = load_model_bf16()
    data = load_data()
    latencies = []
    tokens_per_second = []
    text_responses = []
    warmup = model.generate(**tokenizer(data["prompt"].iloc[0], 
                                        return_tensors="pt").to(device), max_new_tokens=1)
    data = data.head(N_SAMPLES)
    for prompt in data["prompt"].to_list():
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        latency_start_time = time.perf_counter()
        single_output = model.generate(**tokens, max_new_tokens=1)
        latency = time.perf_counter() - latency_start_time
        latencies.append(latency)
        token_response = model.generate(**tokens, max_new_tokens= 20, eos_token_id=None,
                                    early_stopping=False)
        full_gen_start_time = time.perf_counter()
        full_output = model.generate(**tokens, max_new_tokens= THROUGHPUT_TOKENS+1, eos_token_id=None,
                                    early_stopping=False)
        full_gen_time = time.perf_counter() - full_gen_start_time
        tokens_per_second.append(THROUGHPUT_TOKENS / (full_gen_time - latency))
        prompt_ntokens = tokens["input_ids"].size(1)
        text_response = tokenizer.decode(token_response[0, prompt_ntokens:], skip_special_tokens=True)
        text_responses.append(text_response)
    data["response"] = text_responses
    data["latency"] = latencies
    data["tokens_per_second"] = tokens_per_second
    plt.figure()
    plt.hist(data["latency"], bins=25)
    plt.ylabel("Number of prompts")
    plt.xlabel("Latency (s)")
    plt.title(f"Latencies (time to first token) for {MODEL}")
    plt.savefig("latencies.png")
    
    plt.figure()
    plt.ylabel("Number of prompts")
    plt.hist(data["tokens_per_second"], bins=25)
    plt.xlabel("Tokens per second")
    plt.title(f"Tokens per second throughput for {MODEL}")
    plt.savefig("tokens_per_sec.png")
    data.to_parquet("bf16_result.parquet")

def get_bf16_pruned_responses():
    N_SAMPLES = 100
    for compression_ratio in [0.001, 0.002, 0.003, 0.004, 0.005]:
        model, tokenizer = load_bf16_model_structured_pruning(compression_ratio)
        data = load_data()
        text_responses = []
        data = data.head(N_SAMPLES)
        for prompt in data["prompt"].to_list():
            tokens = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_ntokens = tokens["input_ids"].size(1)
            token_response = model.generate(**tokens, max_new_tokens= 20, eos_token_id=None,
                                        early_stopping=False)
            text_response = tokenizer.decode(token_response[0, prompt_ntokens:], skip_special_tokens=True)
            text_responses.append(text_response)
        data["response"] = text_responses
        data.to_parquet(f"bf16_result_structured_prune_{compression_ratio}.parquet")
        print(f"done with {compression_ratio}")

def extract_llm_answer(response: str) -> str | None:
    pattern = re.compile(
        r'(?:'                      
          r'answer:\s*([abcd])'     
        r'|'                        
          r'answer\s+is\s+([abcd])' 
        r'|'                        
          r'([abcd])\)'            
        r')',
        re.IGNORECASE
    )
    m = pattern.search(response)
    if not m:
        return ""
    extracted_answer = next(g for g in m.groups() if g).lower()
    if extracted_answer is not None:
        return extracted_answer
    return ""

def eval_llm_accuracy (df: pd.DataFrame) -> tuple[float, float]:
    df["llm_answer"] = df["response"].apply(extract_llm_answer)
    df["answer"] = df["answer"].str.lower()
    llm_ans_count = (df["llm_answer"] != "").sum()
    llm_correct_ans_count = (df["llm_answer"] == df["answer"]).sum()
    acc = llm_correct_ans_count/llm_ans_count
    prop_answered = llm_ans_count / len(df)
    return acc, prop_answered

def plot_llm_accuracies ():
    dfs = [pd.read_parquet("bf16_result.parquet")]
    for compression_ratio in [0.001, 0.002, 0.003, 0.004, 0.005]:
        dfs.append(pd.read_parquet(f"bf16_result_structured_prune_{compression_ratio}.parquet"))
    all_compression_ratios = [0, 0.001, 0.002, 0.003, 0.004, 0.005]
    all_accs = []
    all_prop_answered = []
    for df in dfs:
        acc, prop_answered = eval_llm_accuracy(df)
        all_accs.append(acc)
        all_prop_answered.append(prop_answered)
    result_df = pd.DataFrame({"compression_ratio": all_compression_ratios,
                              "accuracy": all_accs,
                              "prop_answered": all_prop_answered})
    result_df.to_parquet("bf16_acc.parquet")
    plt.figure(figsize=(9,6))
    plt.plot(result_df["compression_ratio"].to_numpy() * 100, result_df["accuracy"].to_numpy() * 100)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("% of weights pruned (%)")
    plt.title(f"{MODEL} BF16: accuracy vs. structured pruning compression ratio")
    plt.savefig("pruning_acc.png")
    
    plt.figure(figsize=(12,8))
    plt.plot(result_df["compression_ratio"].to_numpy() * 100, result_df["prop_answered"].to_numpy() * 100)
    plt.ylabel("Percentage of questions clearly answered")
    plt.xlabel("% of weights pruned (%)")
    plt.title(f"{MODEL} BF16: % questions clearly answered vs. structured pruning compression ratio")
    plt.savefig("prop_answered.png")

run_bf16_exp()
get_bf16_pruned_responses()
plot_llm_accuracies()
