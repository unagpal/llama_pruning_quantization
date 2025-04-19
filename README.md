# llama_pruning_quantization

Dataset: RACE-H test set (reading comprehension): https://huggingface.co/datasets/ehovy/race

cache_llama_models.py: Retrieve LLAMA models from Hugging Face and cache to enhance speed of future model loads

cache_opt_models.py: Retrieve OPT models from Hugging Face and cache to enhance speed of future model loads

eval_llama_models.py: Evaluate throughput, latency, and accuracy of LLAMA models with different pruning/quantization approaches

eval_opt_models.py: Evaluate throughput, latency, and accuracy of OPT models with different pruning/quantization approaches
