import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import re
from transformers import BitsAndBytesConfig, LlamaForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity
from torch.nn.utils import prune
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from race import load_data
from llama import load_model_bf16_bmk, load_model_bf16
from pruning import L1PrunedInputLinear, L1PrunedOutputLinear

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
Leverage custom L1 pruned linear layers to load BF16 Llama model and 
prune linear layers, shrinking size of matmuls
"""
def load_bf16_model_l1_structured_pruning_shrink(compression_ratio: float, prune_type: str):
    model, tokenizer = load_model_bf16()

    # Recursively replace every Linear layer with L1 pruned input linear
    def _prune_linear(model: nn.Module, prune_type: str):
        for name, child in list(model.named_children()):
            if isinstance(child, nn.Linear):
                if prune_type == "input":
                    setattr(model, name, L1PrunedInputLinear(child, compression_ratio))
                elif prune_type == "output":
                    setattr(model, name, L1PrunedOutputLinear(child, compression_ratio))
                else:
                    raise ValueError("Only prune_type 'input' and 'output' are supported")
            else:
                _prune_linear(child, prune_type)

    _prune_linear(model, prune_type)
    model.to(device)
    model.eval()
    return model, tokenizer

# For each BF16 pruning configuration, measure latency (time to first token) and throughput (tokens per second)
def run_bf16_prune_perf_exp(mode: str, compression_ratio: float) -> pd.DataFrame:
    THROUGHPUT_TOKENS = 100
    N_SAMPLES = 10
    if mode == "normal":
        model, tokenizer = load_model_bf16_bmk()
    elif mode == "l1_zeros_input":
        model, tokenizer = load_bf16_model_structured_pruning_zeros(compression_ratio, "input", norm_type = 1)
    elif mode == "l1_zeros_output":
        model, tokenizer = load_bf16_model_structured_pruning_zeros(compression_ratio, "output", norm_type = 1)
    elif mode == "l1_pruned_input":
        model, tokenizer = load_bf16_model_l1_structured_pruning_shrink(compression_ratio, "input")
    elif mode == "l1_pruned_output":
        model, tokenizer = load_bf16_model_l1_structured_pruning_shrink(compression_ratio, "output")
    else:
        raise ValueError("mode must be one of: 'normal', 'l1_zeros_input', 'l1_zeros_output', 'l1_pruned_input', 'l1_pruned_output'")

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
    data["compression_ratio"] = compression_ratio
    return data

# Iterate through various BF16 compression ratios and measure impact on inference efficiency
def run_all_bf16_prune_perf_exp() -> pd.DataFrame:
    out_dfs = []
    compression_ratios = [0.001, 0.01, 0.1, 0.5]
    for compression_ratio in compression_ratios:
        for mode in ["normal", "l1_zeros_input", "l1_zeros_output", "l1_pruned_input", "l1_pruned_output"]:
            out_dfs.append(run_bf16_prune_perf_exp(mode, compression_ratio))
    df = pd.concat(out_dfs)
    df.to_parquet("bf16_prune_perf.parquet")

# For each pruning configuration, measure memory usage using Pytorch profiling
def run_bf16_prune_mem_exp(mode: str, compression_ratio: float) -> pd.DataFrame:
    THROUGHPUT_TOKENS = 100
    N_SAMPLES = 5
    if mode == "normal":
        model, tokenizer = load_model_bf16_bmk()
    elif mode == "l1_zeros_input":
        model, tokenizer = load_bf16_model_structured_pruning_zeros(compression_ratio, "input", norm_type = 1)
    elif mode == "l1_zeros_output":
        model, tokenizer = load_bf16_model_structured_pruning_zeros(compression_ratio, "output", norm_type = 1)
    elif mode == "l1_pruned_input":
        model, tokenizer = load_bf16_model_l1_structured_pruning_shrink(compression_ratio, "input")
    elif mode == "l1_pruned_output":
        model, tokenizer = load_bf16_model_l1_structured_pruning_shrink(compression_ratio, "output")
    else:
        raise ValueError("mode must be one of: 'normal', 'l1_zeros_input', 'l1_zeros_output', 'l1_pruned_input', 'l1_pruned_output'")

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
    data["compression_ratio"] = compression_ratio
    return data

# Evaluate memory usage of pruned models for several BF16 compression ratios
def run_all_bf16_prune_mem_exp() -> pd.DataFrame:
    out_dfs = []
    compression_ratios = [0.001, 0.01, 0.1, 0.5]
    for compression_ratio in compression_ratios:
        for mode in ["normal", "l1_zeros_input", "l1_zeros_output", "l1_pruned_input", "l1_pruned_output"]:
            out_dfs.append(run_bf16_prune_mem_exp(mode, compression_ratio))
    df = pd.concat(out_dfs)
    df.to_parquet("bf16_prune_mem.parquet")

# Plot all latency and throughput results
def plot_all_prune_perf() -> None:
    df = pd.read_parquet("bf16_prune_perf.parquet")
    std_latency = df.loc[df["mode"] == "normal"]["latency"].mean()
    std_tps = df.loc[df["mode"] == "normal"]["tokens_per_second"].mean()
    prune_df = df.loc[df["mode"] != "normal"]
    prune_avg = prune_df.groupby(by=["mode", "compression_ratio"]).agg({"latency": "mean", "tokens_per_second": "mean"}).reset_index()
    compression_ratios = df["compression_ratio"].unique()
    
    # Latency vs. compression ratio
    plt.figure()
    plt.plot(compression_ratios, [std_latency] * len(compression_ratios), label="standard_inference")
    for mode in prune_avg["mode"].unique():
        mode_results = prune_avg.loc[prune_avg["mode"] == mode].sort_values("compression_ratio")
        plt.plot(mode_results["compression_ratio"], mode_results["latency"], label = mode)
    plt.xscale("log")
    plt.xlabel("Proportion of weights pruned (log scale)")
    plt.ylabel("Latency (seconds)")
    plt.title("Latency vs. pruning compression ratio")
    plt.legend()
    plt.savefig("prune_latency.png")
    
    # Tokens per second vs. compression ratio
    plt.figure()
    plt.plot(compression_ratios, [std_tps] * len(compression_ratios), label="standard_inference")
    for mode in prune_avg["mode"].unique():
        mode_results = prune_avg.loc[prune_avg["mode"] == mode].sort_values("compression_ratio")
        plt.plot(mode_results["compression_ratio"], mode_results["tokens_per_second"], label = mode)
    plt.xscale("log")
    plt.xlabel("Proportion of weights pruned (log scale)")
    plt.ylabel("Tokens per second")
    plt.title("Tokens per second vs. proportion of weights pruned")
    plt.legend()
    plt.savefig("prune_tps.png")
    
    # Double bar chart: avg latency and avg tokens per second for standard inference vs. 0.1% compression
    PLOT_COMPRESSION_RATIO = 0.001
    all_latencies = [df.loc[df["mode"] == "normal", "latency"].mean()]
    all_tps = [df.loc[df["mode"] == "normal", "tokens_per_second"].mean()]
    for mode in prune_df["mode"].unique():
        all_latencies.append(prune_df.loc[(prune_df["mode"] == mode) & (prune_df["compression_ratio"] == PLOT_COMPRESSION_RATIO), "latency"].mean())
        all_tps.append(prune_df.loc[(prune_df["mode"] == mode) & (prune_df["compression_ratio"] == PLOT_COMPRESSION_RATIO), "tokens_per_second"].mean())
    categories = ["standard_inference"] + prune_df["mode"].unique().tolist()
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(8, 5))
    lat_bars = ax1.bar(np.arange(len(categories)) - width/2, all_latencies, width, label='Latency', color='C0')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Average Latency (seconds)', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax2 = ax1.twinx()
    tps_bars = ax2.bar(np.arange(len(categories)) + width/2, all_tps, width, label='Throughput', color='C1')
    ax2.set_ylabel('Average Throughput (tokens per second)', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_xticklabels(categories)
    plt.title('Latency and throughput comparison: Llama-3.2-3B 0.1% Pruning')
    plt.tight_layout()
    plt.savefig("prune_latency_throughput_bar.png")

# Plot pruning memory usage impact results
def plot_all_prune_mem() -> None:
    df = pd.read_parquet("bf16_prune_mem.parquet")
    std_peak = df.loc[df["mode"] == "normal"]["peak_bytes"].mean() / (1e9)
    prune_df = df.loc[df["mode"] != "normal"]
    prune_avg = prune_df.groupby(by=["mode", "compression_ratio"]).agg({"peak_bytes": "mean", "max_op_bytes": "mean"}).reset_index()
    prune_avg["peak_bytes"] = prune_avg["peak_bytes"] / (1e9)
    compression_ratios = df["compression_ratio"].unique()
    
    # Peak mem usage vs. compression ratio
    plt.figure(figsize=(9,6))
    plt.plot(compression_ratios, [std_peak] * len(compression_ratios), label="standard_inference")
    for i, mode in enumerate(prune_avg["mode"].unique()):
        mode_results = prune_avg.loc[prune_avg["mode"] == mode].sort_values("compression_ratio")
        # Shift x axis very slightly to visually clarify which lines overlap
        eps = (i - 3/2) * 0.028 * mode_results["compression_ratio"]
        plt.plot(mode_results["compression_ratio"] + eps, mode_results["peak_bytes"], label = mode)
    plt.xscale("log")
    plt.xlabel("Proportion of weights pruned (log scale)")
    plt.ylabel("Peak GB allocated")
    plt.title("Peak GB allocated vs. pruning compression ratio")
    plt.legend()
    plt.savefig("prune_mem_usage.png")

# Run all BF16 Llama pruning experiments, measuring latency/throughput/memory impact
def run_all_exp() -> None:
    run_all_bf16_prune_perf_exp()
    run_all_bf16_prune_mem_exp()
    plot_all_prune_perf()
    plot_all_prune_mem()
