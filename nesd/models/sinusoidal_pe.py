import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPE(nn.Module):
    r"""Sinusodial positional embedder.
    """

    def __init__(
        self, 
        dim: int = 256, 
        max_period: int = 10000,
        scale: float = 1.0
    ):
        super().__init__()

        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)  # (1./10000)**(2i/d)
        self.register_buffer("freqs", freqs)

        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        r"""Calculate position embedding.

        b: batch_size
        d: dim

        Args:
            x: (any,), between 0. and 1.

        Outputs:
            out: (any, d)
        """
        
        x = self.scale * x[..., None] * self.freqs  # (b, d/2)
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)  # (b, d)
        
        return x
