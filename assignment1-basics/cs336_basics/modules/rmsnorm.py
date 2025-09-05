import torch
import torch.nn as nn

from einops import reduce

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.gains = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32) # Cast to keep high precision

        var = reduce(x.pow(2), "... d_model -> ... 1", "mean")

        # multiply by sqrt inverse is faster
        x_norm = x * torch.rsqrt(var + self.eps) 
        
        result = x_norm * self.gains

        return result.to(in_dtype)
