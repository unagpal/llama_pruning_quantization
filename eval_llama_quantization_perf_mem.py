
from llama import load_model_bf16_bmk
from race import load_data
from gptq import load_llama_gptq
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
import matplotlib.pyplot as plt
from pruning import L1PrunedInputLinear

device = "cuda" if torch.cuda.is_available() else "cpu" 

"""
Prune GPTQ-quantized LLAMA models the maximum possible while retaining
at least 50% RACE-H reading comprehension accuracy
"""
def load_optimal_quantized_model_l1_structured_pruning_shrink(nbits: int):
    # Maximum compression ratios that remain above 50% accuracy by nbits
    PRUNE_TYPE = "input"
    OPTIMIZED_COMPRESSION_RATIOS = {4: 0.1, 8: 0.35}
    compression_ratio = OPTIMIZED_COMPRESSION_RATIOS[nbits]
    model, tokenizer = load_llama_gptq(nbits)

    # Recursively replace every Linear layer with L1 pruned input linear
    def _prune_linear_input(model: nn.Module):
        for name, child in list(model.named_children()):
            if isinstance(child, nn.Linear):
                setattr(model, name, L1PrunedInputLinear(child, compression_ratio))
            else:
                _prune_linear_input(child)

    _prune_linear_input(model)
    model.to(device)
    model.eval()
    return model, tokenizer

# Measure latency (time to first token) and throughput (tokens/second) for quantized models
def run_quantization_perf_exp(mode: str) -> pd.DataFrame:
    THROUGHPUT_TOKENS = 100
    N_SAMPLES = 20
    if mode == "normal":
        model, tokenizer = load_model_bf16_bmk()
    elif mode == "8bit":
        model, tokenizer = load_llama_gptq(8)
    elif mode == "4bit":
        model, tokenizer = load_llama_gptq(4)
    elif mode == "8bitpruned":
        model, tokenizer = load_optimal_quantized_model_l1_structured_pruning_shrink(8)
    elif mode == "4bitpruned":
        model, tokenizer = load_optimal_quantized_model_l1_structured_pruning_shrink(4)
    else:
        raise ValueError("mode must be one of: 'normal', '8bit', '8bitpruned', '4bit', '4bitpruned'")

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
    data["mode"] = mode
    return data

# Cache quantized model efficiency for various quantization and quantization + pruning setups
def run_all_quantization_perf_exp() -> None:
    out_dfs = []
    for mode in ["normal", "8bitpruned", "4bitpruned", "8bit", "4bit"]:
        out_dfs.append(run_quantization_perf_exp(mode))
    df = pd.concat(out_dfs)
    df.to_parquet("quant_perf.parquet")

# Measure peak memory allocated for quantized model inference
def run_quantization_mem_exp(mode: str) -> pd.DataFrame:
    THROUGHPUT_TOKENS = 100
    N_SAMPLES = 5
    if mode == "normal":
        model, tokenizer = load_model_bf16_bmk()
    elif mode == "8bit":
        model, tokenizer = load_llama_gptq(8)
    elif mode == "4bit":
        model, tokenizer = load_llama_gptq(4)
    elif mode == "8bitpruned":
        model, tokenizer = load_optimal_quantized_model_l1_structured_pruning_shrink(8)
    elif mode == "4bitpruned":
        model, tokenizer = load_optimal_quantized_model_l1_structured_pruning_shrink(4)
    else:
        raise ValueError("mode must be one of: 'normal', '8bit', '8bitpruned', '4bit', '4bitpruned'")

    data = load_data()
    peak_alloc_bytes = []
    max_bytes = []
    std_bytes = []
    prompt_token_lengths = []
    warmup = model.generate(**tokenizer(data["prompt"].iloc[0], 
                                        return_tensors="pt").to(device), max_new_tokens=1)
    data = data.head(N_SAMPLES)
    for prompt in data["prompt"].to_list():
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        with profile(activities = [ProfilerActivity.CUDA], profile_memory=True, record_shapes=False) as pt_profile:
            _ = model.generate(**tokens, max_new_tokens= THROUGHPUT_TOKENS+1, eos_token_id=None,
                                    early_stopping=False)
            peak_bytes = torch.cuda.max_memory_allocated()

        prof_avgs = pt_profile.key_averages()
        prompt_ntokens = tokens["input_ids"].size(1)
        mem_usages = pd.Series([event.device_memory_usage for event in prof_avgs if event.device_memory_usage])
        peak_alloc_bytes.append(peak_bytes)
        max_bytes.append(mem_usages.max())
        std_bytes.append(mem_usages.std())
        prompt_token_lengths.append(prompt_ntokens)
    
    data["max_op_bytes"] = max_bytes
    data["std_op_bytes"] = std_bytes
    data["peak_bytes"] = peak_alloc_bytes
    data["prompt_token_length"] = prompt_token_lengths
    data["mode"] = mode
    return data

# Cache peak Llama inference memory usage for various quantization and quantization + pruning setups
def run_all_quantization_mem_exp() -> None:
    out_dfs = []
    for mode in ["normal", "8bitpruned", "4bitpruned", "8bit", "4bit"]:
        out_dfs.append(run_quantization_mem_exp(mode))
    df = pd.concat(out_dfs)
    df.to_parquet("quant_mem.parquet")

# Create triple bar chart showing various percentiles of inference latency/throughput/memory usage
def triple_bar_quant_chart (stats: pd.DataFrame, title: str, ylabel: str, filename: str) -> None:
    modes = stats.index.tolist()
    percentiles = [float(prctile) for prctile in stats.columns]
    labels = [f"{int(prctile * 100)}th percentile" if prctile != 0.5 else "Median" for prctile in percentiles]
    x = np.arange(len(stats))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12,8))

    for group_idx, (prctile, label) in enumerate(zip(percentiles, labels)):
        ax.bar(x + (group_idx - 1) * width, stats[prctile].to_numpy(), width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{filename}.png")

# Illustrate impact of quantization on latency, throughput, and memory usage
def plot_quantization_exp_results () -> None:
    df = pd.read_parquet("quant_perf.parquet")
    df.loc[df["mode"] == "normal", "mode"] = "bf16"
    latency_stats = df.groupby("mode")["latency"].quantile([0.2, 0.5, 0.8]).unstack()
    tps_stats = df.groupby("mode")["tokens_per_second"].quantile([0.2, 0.5, 0.8]).unstack()
    triple_bar_quant_chart(latency_stats, "GPTQ quantization and pruning impact on Llama-3.2-3B latency (time to first token) distribution", "Latency (seconds)", "quant_latency")
    triple_bar_quant_chart(tps_stats, "GPTQ quantization and pruning impact on Llama-3.2-3B throughput (tokens per second) distribution", "Throughput (tokens per second)", "quant_tps")
    
    df = pd.read_parquet("quant_mem.parquet")
    df.loc[df["mode"] == "normal", "mode"] = "bf16"
    df["peak_bytes"] = df["peak_bytes"] / (1e9)
    mem_stats = df.groupby("mode")["peak_bytes"].quantile([0.2, 0.5, 0.8]).unstack()
    triple_bar_quant_chart(mem_stats, "GPTQ quantization impact on Llama-3.2-3B peak inference memory allocation (GB) distribution", "Peak inference memory usage (GB)", "quant_mem_usage")

# Run all quantization and quantization + pruning latency, throughput, and mmemory usage experiments
def run_all_experiments() -> None:
    run_all_quantization_mem_exp()
    run_all_quantization_perf_exp()
    plot_quantization_exp_results()
