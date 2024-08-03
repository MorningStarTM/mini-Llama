import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 32
    n_heads: int = 3
    n_kv_heads: Optional[int] = None
    vocab_size:int = -1
    multiple_of: int = 256
    norm_eps: float = 1e-5
    ffn_dim_multiplier = Optional[float] = None

    max_batch_size: int = 32
    max_seq_len: int = 512
    device: str = None

    