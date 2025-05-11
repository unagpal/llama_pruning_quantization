
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from torch.nn.utils import prune
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Custom linear layer with L1 structured input pruning that shrinks 
size of matmul. Used because torch ln_structured pruning does not 
reduce matmul size.
"""
class L1PrunedInputLinear (nn.Module):
    def __init__(self, linear: nn.Linear, compression_ratio: float):
        super().__init__()
        W = linear.weight.data.clone()
        b = linear.bias.data.clone() if linear.bias else None

        # L1 pruning along input dim
        in_features = W.size(1)
        kept_features = max(in_features - int(in_features * compression_ratio), 1)
        kept_idx = torch.topk(W.abs().sum(dim=0), kept_features).indices.sort().values
        self.register_buffer("kept_idx", kept_idx)

        # Create smaller contiguous weight matrix
        self.register_buffer("W_pruned", W[:, kept_idx].contiguous())
        self.register_buffer("b_pruned", b.contiguous() if b is not None else None)

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        # Prune input for smaller matmul
        pruned_input = x.index_select(-1, self.kept_idx)
        return F.linear(pruned_input, self.W_pruned, self.b_pruned)

"""
Custom linear layer with L1 structured output pruning that shrinks
size of matmul. Used because torch ln_structured pruning does not 
reduce matmul size.
"""
class L1PrunedOutputLinear (nn.Module):
    def __init__(self, linear: nn.Linear, compression_ratio: float):
        super().__init__()
        W = linear.weight.data.clone()
        b = linear.bias.data.clone() if linear.bias else None

        # L1 pruning along output dim
        out_features = W.size(0)
        kept_features = max(out_features - int(out_features * compression_ratio), 1)
        kept_idx = torch.topk(W.abs().sum(dim=1), kept_features).indices.sort().values

        # Create smaller contiguous weight matrix
        self.register_buffer("kept_idx", kept_idx)
        self.register_buffer("W_pruned", W[kept_idx, :].contiguous())
        self.register_buffer("b_pruned", b.contiguous() if b is not None else None)
        self.output_dim = out_features

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(list(x.shape[:-1]) + [self.output_dim], dtype=torch.bfloat16).to(device)
        output[..., self.kept_idx] = F.linear(x, self.W_pruned, self.b_pruned)
        return output
