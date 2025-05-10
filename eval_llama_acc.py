import pandas as pd
import numpy as np
import re
from transformers import BitsAndBytesConfig, LlamaForCausalLM, AutoTokenizer
from gptq import load_llama_gptq
import torch
from torch import nn
from torch.nn.utils import prune
import time
import logging
import matplotlib.pyplot as plt
from race import load_data
from llama import load_model_bf16

MODEL = "meta-llama/Llama-3.2-3B"
MODEL_PATH = "./llama_models/"
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Standard L1 structured pruning for linear layers: masks pruned weights to 0,
does not shrink matmul size
"""
def load_bf16_model_structured_pruning_zeros(compression_ratio: float, prune_type: str, norm_type: int = 1):
    if prune_type == "input":
        prune_dim = 1
    elif prune_type == "output":
        prune_dim = 0
    else:
        raise ValueError("prune_type must be either 'input' or 'output'")

    model, tokenizer = load_model_bf16()
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            prune.ln_structured(
                layer,
                name="weight",
                amount=compression_ratio, 
                n=norm_type,
                dim=prune_dim
            )
            prune.remove(layer, "weight")

    model.to(device)
    model.eval()
    return model, tokenizer

"""
Standard L1 structured pruning for linear layers: masks pruned weights to 0,
does not shrink matmul size
"""
def load_quantized_model_structured_pruning_zeros(compression_ratio: float, prune_type: str, nbits: int, norm_type: int = 1):
    if prune_type == "input":
        prune_dim = 1
    elif prune_type == "output":
        prune_dim = 0
    else:
        raise ValueError("prune_type must be either 'input' or 'output'")

    model, tokenizer = load_llama_gptq(nbits)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            prune.ln_structured(
                layer,
                name="weight",
                amount=compression_ratio, 
                n=norm_type,
                dim=prune_dim
            )
            prune.remove(layer, "weight")

    model.to(device)
    model.eval()
    return model, tokenizer

def run_bf16_exp() -> None:
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
        torch.cuda.synchronize()
        latency_start_time = time.perf_counter()
        single_output = model.generate(**tokens, max_new_tokens=1)
        torch.cuda.synchronize()
        latency = time.perf_counter() - latency_start_time
        latencies.append(latency)
        token_response = model.generate(**tokens, max_new_tokens= 20, eos_token_id=None,
                                    early_stopping=False)
        torch.cuda.synchronize()
        full_gen_start_time = time.perf_counter()
        full_output = model.generate(**tokens, max_new_tokens= THROUGHPUT_TOKENS+1, eos_token_id=None,
                                    early_stopping=False)
        torch.cuda.synchronize()
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

def cache_gptq_responses() -> None:
    THROUGHPUT_TOKENS = 100
    N_SAMPLES = 100
    for nbits in [4, 8]:
        data = load_data()
        model, tokenizer = load_llama_gptq(nbits)
        text_responses = []
        warmup = model.generate(**tokenizer(data["prompt"].iloc[0], 
                                            return_tensors="pt").to(device), max_new_tokens=1)
        data = data.head(N_SAMPLES)
        for prompt in data["prompt"].to_list():
            tokens = tokenizer(prompt, return_tensors="pt").to(device)
            token_response = model.generate(**tokens, max_new_tokens= 20, eos_token_id=None,
                                        early_stopping=False)
            prompt_ntokens = tokens["input_ids"].size(1)
            text_response = tokenizer.decode(token_response[0, prompt_ntokens:], skip_special_tokens=True)
            text_responses.append(text_response)
        data["response"] = text_responses
        data.to_parquet(f"{nbits}bit_result_responses.parquet")

def get_pruned_responses(nbits: int = 16, high_compression: bool = False):
    if nbits not in [4, 8, 16]:
        raise ValueError("Only nbits values of 4, 8, and 16 are supported")

    N_SAMPLES = 100
    if high_compression:
        compression_ratios = [0.01, 0.05, 0.1, 0.25, 0.35, 0.5]
    else:
        compression_ratios = [0.001, 0.002, 0.003, 0.004, 0.005]

    for compression_ratio in compression_ratios:
        for prune_type in ["input", "output"]:
            if nbits == 16:
                model, tokenizer = load_bf16_model_structured_pruning_zeros(compression_ratio, prune_type)
            else:
                model, tokenizer = load_quantized_model_structured_pruning_zeros(compression_ratio, prune_type, nbits)

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

            out_path = f"bf16_result_structured_prune_{prune_type}_{compression_ratio}.parquet" if (
                nbits==16) else f"{nbits}bits_result_structured_prune_{prune_type}_{compression_ratio}.parquet"
            data.to_parquet(out_path)
            print(f"done with {compression_ratio},{prune_type}")

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

def plot_pruning_accuracies ():
    dfs = [pd.read_parquet("bf16_result.parquet")] * 2 + [pd.read_parquet(
        "8bit_result_responses.parquet")] * 2 + [pd.read_parquet("4bit_result_responses.parquet")] * 2
    all_compression_ratios = [0] * 6
    all_prune_types = ["input", "output"] * 3
    all_dtypes = ["bf16"] * 2 + ["8bit"] * 2 + ["4bit"] * 2
    dtype_map = {16: "bf16", 8: "8bit", 4: "4bit"}
    for nbits in [16, 8, 4]:
        for compression_ratio in [0.001, 0.002, 0.003, 0.004, 0.005]:
            for prune_type in ["input", "output"]:
                path = f"bf16_result_structured_prune_{prune_type}_{compression_ratio}.parquet" if (
                    nbits==16) else f"{nbits}bits_result_structured_prune_{prune_type}_{compression_ratio}.parquet"
                dfs.append(pd.read_parquet(path))
                all_compression_ratios.append(compression_ratio)
                all_prune_types.append(prune_type)
                all_dtypes.append(dtype_map[nbits])

    all_accs = []
    all_prop_answered = []
    for df in dfs:
        acc, prop_answered = eval_llm_accuracy(df)
        all_accs.append(acc)
        all_prop_answered.append(prop_answered)
    result_df = pd.DataFrame({"compression_ratio": all_compression_ratios,
                              "prune_type": all_prune_types,
                              "dtype": all_dtypes,
                              "accuracy": all_accs,
                              "prop_answered": all_prop_answered})
    result_df.to_parquet("all_acc.parquet")
    input_prune_dfs = []
    output_prune_dfs = []
    dtypes = ["bf16", "8bit", "4bit"]
    for dtype in dtypes:
        input_prune_dfs.append(result_df.loc[(result_df["prune_type"] == "input") & (result_df["dtype"] == dtype)])
        output_prune_dfs.append(result_df.loc[(result_df["prune_type"] == "output") & (result_df["dtype"] == dtype)])
    
    plt.figure(figsize=(9,6))
    for dtype_idx, dtype in enumerate(dtypes):
        input_prune_df = input_prune_dfs[dtype_idx]
        output_prune_df = output_prune_dfs[dtype_idx]
        plt.plot(input_prune_df["compression_ratio"].to_numpy() * 100, input_prune_df["accuracy"].to_numpy() * 100, label=f"{dtype}_l1_pruned_input")
        plt.plot(output_prune_df["compression_ratio"].to_numpy() * 100, output_prune_df["accuracy"].to_numpy() * 100, label=f"{dtype}_l1_pruned_output")

    plt.ylabel("Accuracy (%)")
    plt.xlabel("% of linear weights pruned (%)")
    plt.title(f"{MODEL}: accuracy vs. structured pruning compression ratio")
    plt.legend()
    plt.savefig("all_pruning_acc.png")
    
    plt.figure(figsize=(12,8))
    for dtype_idx, dtype in enumerate(dtypes):
        input_prune_df = input_prune_dfs[dtype_idx]
        output_prune_df = output_prune_dfs[dtype_idx]
        plt.plot(input_prune_df["compression_ratio"].to_numpy() * 100, input_prune_df["prop_answered"].to_numpy() * 100, label=f"{dtype}_l1_pruned_input")
        plt.plot(output_prune_df["compression_ratio"].to_numpy() * 100, output_prune_df["prop_answered"].to_numpy() * 100, label=f"{dtype}_l1_pruned_output")

    plt.ylabel("Percentage of questions clearly answered")
    plt.xlabel("% of weights pruned (%)")
    plt.title(f"{MODEL}: % questions clearly answered vs. structured pruning compression ratio")
    plt.legend()
    plt.savefig("all_prop_answered.png")

# Plot accuracy for GPTQ quantization followed by pruning
def plot_quant_pruning_accuracies ():
    dfs = [pd.read_parquet("8bit_result_responses.parquet")] * 2 + [pd.read_parquet("4bit_result_responses.parquet")] * 2
    all_compression_ratios = [0] * 4
    all_prune_types = ["input", "output"] * 2
    all_dtypes = ["8bit"] * 2 + ["4bit"] * 2
    dtype_map = {8: "8bit", 4: "4bit"}
    for nbits in [8, 4]:
        for compression_ratio in [0.005, 0.01, 0.05, 0.1, 0.25, 0.35, 0.5]:
            for prune_type in ["input", "output"]:
                path = f"{nbits}bits_result_structured_prune_{prune_type}_{compression_ratio}.parquet"
                dfs.append(pd.read_parquet(path))
                all_compression_ratios.append(compression_ratio)
                all_prune_types.append(prune_type)
                all_dtypes.append(dtype_map[nbits])

    all_accs = []
    all_prop_answered = []
    for df in dfs:
        acc, prop_answered = eval_llm_accuracy(df)
        all_accs.append(acc)
        all_prop_answered.append(prop_answered)
    result_df = pd.DataFrame({"compression_ratio": all_compression_ratios,
                              "prune_type": all_prune_types,
                              "dtype": all_dtypes,
                              "accuracy": all_accs,
                              "prop_answered": all_prop_answered})
    input_prune_dfs = []
    output_prune_dfs = []
    dtypes = ["8bit", "4bit"]
    for dtype in dtypes:
        input_prune_dfs.append(result_df.loc[(result_df["prune_type"] == "input") & (result_df["dtype"] == dtype)])
        output_prune_dfs.append(result_df.loc[(result_df["prune_type"] == "output") & (result_df["dtype"] == dtype)])
    
    plt.figure(figsize=(12,8))
    for dtype_idx, dtype in enumerate(dtypes):
        input_prune_df = input_prune_dfs[dtype_idx]
        output_prune_df = output_prune_dfs[dtype_idx]
        plt.plot(input_prune_df["compression_ratio"].to_numpy() * 100, input_prune_df["accuracy"].to_numpy() * 100, label=f"{dtype}_l1_pruned_input")
        plt.plot(output_prune_df["compression_ratio"].to_numpy() * 100, output_prune_df["accuracy"].to_numpy() * 100, label=f"{dtype}_l1_pruned_output")

    plt.ylabel("Accuracy (%)")
    plt.xlabel("% of linear weights pruned (%)")
    plt.title(f"{MODEL}: accuracy vs. structured pruning compression ratio for GPTQ quantized models")
    plt.legend()
    plt.savefig("quant_pruning_acc.png")

def plot_quantization_accuracy() -> None:
    bf16 = pd.read_parquet("bf16_result.parquet")
    gptq8 = pd.read_parquet("8bit_result_responses.parquet")
    gptq4 = pd.read_parquet("4bit_result_responses.parquet")
    bf16_acc, bf16_prop_ans = eval_llm_accuracy(bf16)
    gptq8_acc, gptq8_prop_ans = eval_llm_accuracy(gptq8)
    gptq4_acc, gptq4_prop_ans = eval_llm_accuracy(gptq4)
    groups = ["bf16", "8bit", "4bit"]
    acc = [bf16_acc * 100, gptq8_acc * 100, gptq4_acc * 100]
    
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x, acc, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Llama-3.2-3B accuracy impact of GPTQ quantization")
    
    plt.tight_layout()
    plt.savefig("quant_acc")

def run_all_exp():
    run_bf16_exp()
    get_pruned_responses()
    get_pruned_responses(8, high_compression=True)
    get_pruned_responses(4, high_compression=True)
    cache_gptq_responses()
    plot_pruning_accuracies()
    plot_quant_pruning_accuracies()
    plot_quantization_accuracy()
