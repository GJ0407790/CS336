import math
import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum, rearrange

from cs336_basics.modules.rope import Rope

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    x: torch.Tensor of any shape
    dim: int dimension to apply the softmax over
    """
    x_max = x.amax(dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)

    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... q d_k"],
    K: Float[Tensor, " ... k d_k"],
    V: Float[Tensor, " ... k d_v"],
    mask: Float[Tensor, "q k"] | None = None,
) -> Float[Tensor, " ... d_v"]:
    inv_dk = 1 / (Q.shape[-1] ** 0.5)

    dot_product = einsum(Q, K, "... q d, ... k d -> ... q k") * inv_dk

    # Add -inf to the masked postions
    if mask is not None:
        dot_product = dot_product.masked_fill(mask == 0, float("-inf"))

    attn = softmax(dot_product, dim=-1) # (..., q, k)
    
    return einsum(attn, V, "... q k, ... k d_v -> ... q d_v")

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: Rope | None=None, device=None, dtype=None):
        """
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        """
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.rope = rope

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        # Combine W_Q, W_K, W_V into a single matrix
        # W_Q (h*d_k, d_model), W_K (h*d_k, d_model), W_V (h*d_v, d_model)
        # the shape will be (h * (2 * d_k + d_v), d_model)
        self.W_qkv = nn.Parameter(
            torch.empty((num_heads * (2 * self.d_k + self.d_v), d_model), device=device, dtype=dtype)
        )

        self.W_o = nn.Parameter(
            torch.empty((d_model, num_heads * self.d_v), device=device, dtype=dtype)
        )

        self._init_weights()
    
    def _init_weights(self):
        """Initializes weights with a truncated normal distribution."""
        qk_std = math.sqrt(2 / (self.d_model + self.h * self.d_k))
        qk_rows = 2 * self.h * self.d_k
        nn.init.trunc_normal_(self.W_qkv[qk_rows:], mean=0.0, std=qk_std, a=-3*qk_std, b=3*qk_std)

        v_std = math.sqrt(2 / (self.d_model + self.h * self.d_v))
        nn.init.trunc_normal_(self.W_qkv[:qk_rows], mean=0.0, std=v_std, a=-3*v_std, b=3*v_std)

        out_std = math.sqrt(2 / (self.d_model + self.h * self.d_v))
        nn.init.trunc_normal_(self.W_o, mean=0.0, std=out_std, a=-3*out_std, b=3*out_std)

    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_model"], 
        token_positions: Int[Tensor, " ... sequence_length"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        # Project input to queries, keys, values spaces
        proj = einsum(x, self.W_qkv, "... seq d_model, hkv d_model -> ... seq hkv")
        Q, K, V = proj.split(self.h * self.d_k, dim=-1)

        # Reshape to multiple heads
        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.h)
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.h)
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.h)

        # Apply rope
        seq_len = x.shape[-2]

        if self.rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Causual mask
        mask = torch.tril(
            torch.ones((seq_len, seq_len), device=x.device)
        )
        mha = scaled_dot_product_attention(Q, K, V, mask)

        # Rearange back for output projection
        mha = rearrange(mha, "... h seq d -> ... seq (h d)")

        return einsum(mha, self.W_o, "... seq hd, d hd -> ... seq d")
