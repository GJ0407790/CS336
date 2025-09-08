import torch

import torch.nn as nn
from einops import einsum, rearrange, repeat

class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self._build_cache(device)

    def _build_cache(self, device):
        # Build the sin and cos cache for RoPE
        # Generate the frequencies
        theta = self.theta
        d_k = self.d_k
        max_seq_len = self.max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # Compute the outer product to get the angles
        inv_freq = repeat(inv_freq, 'd -> (d repeat)', repeat=2)
        angles = torch.outer(pos, inv_freq) # seq 1, 1 d -> seq d

        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor of shape (..., seq_len, d_k)
        token_positions: torch.Tensor of shape (..., seq_len) giving the position ids for each token in the sequence

        return torch.Tensor of shape (..., seq_len, d_k) (same as input x)
        """
        sin = self.get_buffer("sin_cache")[token_positions]
        cos = self.get_buffer("cos_cache")[token_positions]

        x_pair = rearrange(x, '... (d pair) -> ... d pair', pair=2) # split into chunks of 2
        x_pair = x_pair.flip(-1) # swap the pairs

        rotated_x = rearrange(x_pair, '... d pair -> ... (d pair)') # reshape back to x
        rotated_x[..., 0::2] *= -1 # Negate the even indices

        return x * cos + rotated_x * sin
