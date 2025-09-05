import torch
import math

import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)    
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initializes weights with a truncated normal distribution."""
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor of shape (..., d_in)

        returns: torch.Tensor of shape (..., d_out)
        """
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")