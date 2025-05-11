# HPML Project: Quantization and pruning for efficient Llama-3.2-3B inference in reading comprehension tasks

## Team Information
- **Team Name**: Team FastLlama
- **Members**:
  - Udai Nagpal (ugn2000)

---

## 1. Problem Statement

Apply post-training quantization, pruning (without retraining), and quantization followed by pruning (without retraining) to maximize tokens per second, and minimize latency and peak memory usage, of Llama-3.2-3B inference while preserving over 50% accuracy in RACE-H high school reading comprehension questions.

---

## 2. Model Description
Used Llama-3.2-3B, a transformer-based LLM. The PyTorch framework was used.

The following customizations were made to Llama-3.2-3B:
- GPTQ quantization into 4-bit and 8-bit model versions based on the C4 dataset with block size 128
- L1 structured pruning of linear layers (both input and output pruning). To implement L1 structured pruning in a memory-efficient manner, custom linear layers L1PrunedInputLinear and L1PrunedOutputLinear were defined that are equivalent to linear layers with structured input and output pruning respectively, but shrink the size of the matrix multiplication.
- GPTQ quantization into 4-bit and 8-bit model versions followed by L1 structured pruning

---

## 3. Final Results Summary

Summary of performance and accuracy results (where '8 bit pruned' uses 35% linear layer pruning and '4 bit pruned' uses 10% linear layer pruning):

| Model              | Median Latency (ms) | Median Tokens/sec | Peak Memory Allocated (GB) | RACE-H Accuracy |
|--------------------|---------------------|-------------------|----------------------------|-----------------|
| 4 bit              | 25.95               | 43.2              | 2.40                       | 62.6%           |
| 4 bit pruned       | 25.93               | 43.1              | 3.10*                      | 53.3%           |
| 8 bit              | 25.94               | 43.2              | 3.82                       | 62.0%           |
| 8 bit pruned       | 25.95               | 43.5              | 4.33*                      | 52.8%           |
| bf16 (full model)  | 30.78               | 38.1              | 6.56                       | 69.0%           |

*: Lower memory usage for these models (below that of corresponding un-pruned models) should be achievable with custom Cuda kernels for L1 structured linear layer pruning.

Summary of maximum pruning after GPTQ quantization that preserves at least 50% RACE-H accuracy: 

| Quantization Type | L1 Structured Linear Layer Pruning Type | Max Linear Layer Pruning Fraction Maintaining Above 50% Accuracy |
|-------------------|------------------------------------------|------------------------------------------------------------------|
| 8 bit             | Input                                    | 35%                                                              |
| 8 bit             | Output                                   | 1%                                                               |
| 4 bit             | Input                                    | 10%                                                              |
| 4 bit             | Output                                   | 1%                                                               |

Note that without GPTQ quantization, even 0.1% linear layer structured pruning fails to preserve 50% RACE-H accuracy.

---

## 4. Reproducibility Instructions

Device used: 1 H100 GPU with ARM CPUs (Lambda Labs GH200 instance).

### A. Requirements

Inside the llama_pruning_quantization directory, install dependencies as follows:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_reprod.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

---

B. Wandb Dashboard

View training and evaluation metrics here: Wandb Dashboard Link
(Replace with actual link)

---

### C. Specify for Training or For Inference or if Both 

No model training is included in this project: only inference, post-training quantization, and pruning.

To run GPTQ quantization on LLAMA, run the following within virtual environment venv from step A:
```bash
python cache_llama_models.py
python - << 'EOF'
from gptq import cache_gptq_models
cache_gptq_models()
EOF
```

---

### D. Evaluation

To evaluate the throughput, latency, memory usage, and accuracy of the quantized and pruned models, run the following. Note that the commands in part C are a prerequisite.
```bash
python3 -c "from eval_llama_pruning_perf_mem import run_all_exp; run_all_exp()"
python3 -c "from eval_llama_quantization_perf_mem import run_all_experiments; run_all_experiments()"
python3 -c "from eval_llama_acc import run_all_exp; run_all_exp()"
```

---

### E. Quickstart: Minimum Reproducible Result

To reproduce the result tables from the above Final Results Summary, run:

```bash
# Step 1: Set up environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_reprod.txt --extra-index-url https://download.pytorch.org/whl/cu128

# Step 2: Cache Llama model and calculate GPTQ-quantized models
python cache_llama_models.py
python - << 'EOF'
from gptq import cache_gptq_models
cache_gptq_models()
EOF

# Step 3: Run performance and accuracy experiments including quantization + pruning
python3 -c "from eval_llama_quantization_perf_mem import run_all_experiments; run_all_experiments()"
python3 -c "from eval_llama_acc import run_all_exp; run_all_exp()"
```

---

## 5. Notes

Dataset: RACE-H test set (reading comprehension): https://huggingface.co/datasets/ehovy/race

cache_llama_models.py: Retrieve LLAMA models from Hugging Face and cache to enhance speed of future model loads

cache_opt_models.py: Retrieve OPT models from Hugging Face and cache to enhance speed of future model loads

eval_llama_models.py: Evaluate throughput, latency, and accuracy of LLAMA models with different pruning/quantization approaches

eval_opt_models.py: Evaluate throughput, latency, and accuracy of OPT models with different pruning/quantization approaches

TODO add remaining scripts
TODO mentioned where trained models are saved

Contact information: ugnagpal@gmail.com

