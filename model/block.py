import torch
import torch.nn as nn
from typing import Optional
from model.attention import MultiHeadCausalSelfAttention
from model.feedforward import SwiGLU
from model.embedding import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 max_seq_len: int = 5000, rope_theta: float = 10000.0):
        """
        Pre-norm transformer block, modernised to the LLaMA-style recipe:
          - RMSNorm instead of LayerNorm (no bias, RMS-only rescale)
          - RoPE rotary positions inside attention (no additive PE)
          - SwiGLU feed-forward instead of GELU MLP

        Components:
          self.norm1 = RMSNorm(d_model)
          self.attn  = MultiHeadCausalSelfAttention(d_model, n_heads, dropout, ...)
          self.norm2 = RMSNorm(d_model)
          self.ff    = SwiGLU(d_model, d_ff, dropout)
          self.dropout = nn.Dropout(dropout)   # residual dropout
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(
            d_model, n_heads, dropout, max_seq_len=max_seq_len, rope_theta=rope_theta
        )
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                past_kv: Optional[tuple] = None, use_cache: bool = False):
        """
        Pre-norm residual connections:
          x = x + dropout(attn(norm1(x)))
          x = x + dropout(ff(norm2(x)))

        Pre-norm keeps the residual stream un-normalised so gradients flow freely
        through the skip connection — more stable than post-norm in deep nets.

        Returns `x` normally, or `(x, present_kv)` when use_cache=True (for fast
        autoregressive generation). The default training path is unchanged.
        """
        if use_cache:
            attn_out, present = self.attn(
                self.norm1(x), key_padding_mask=key_padding_mask,
                past_kv=past_kv, use_cache=True,
            )
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ff(self.norm2(x)))
            return x, present

        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask=key_padding_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
