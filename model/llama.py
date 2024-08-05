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


def precompute_theta_pos_frequencies(head_dim:int, seq_len:int, device:str, theta:float = 10000.0):
    assert head_dim % 2 == 0 , "Dimension must be divisible by 2"

    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embedding(x:torch.Tensor, freq_complex: torch.Tensor, device:str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freq_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device=device)


class Transformer(nn.Module):
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embedding = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ouput = nn.Linear(args.dim, self.vocab_size, bias=True)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self/args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, token:torch.Tensor, start_pos: int):
        #(B, seg_len)
        batch_size, seq_len = token.shape
        assert seq_len == 1, "only one token at a time can be processed"

        # (B, seq_len) -> (B, Seq_len, Dim)
        h = self.tok_embedding(token)

        # Retrieve the pairss (m, theta) corresponding to the positions [start_pos, start_pos+seq_len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.ouput(h).float()
        return output

        