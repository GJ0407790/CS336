import torch
import math

import torch.nn as nn
from einops import einsum

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        """
        d_model: int final dimension of the input
        d_ff: int final dimension of the hidden dimension
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Parameter(
            torch.empty((d_ff, d_model), device=device, dtype=dtype)    
        )

        self.w2 = nn.Parameter(
            torch.empty((d_model, d_ff), device=device, dtype=dtype)    
        )

        self.w3 = nn.Parameter(
            torch.empty((d_ff, d_model), device=device, dtype=dtype)    
        )

        self._init_weights()

        
    def _init_weights(self):
        std = math.sqrt(2 / (self.d_model + self.d_ff))

        nn.init.trunc_normal_(self.w1, mean=0.0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor of shape (..., d_model)

        y = (SiLU(x @ w1.T) * (x @ w3.T)) @ w2.T

        returns: y torch.Tensor of shape (..., d_model)
        """
        x1 = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        x1 = self.silu(x1)
        
        x3 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")

        return einsum(x1 * x3, self.w2, "... d_ff, d_model d_ff -> ... d_model")
