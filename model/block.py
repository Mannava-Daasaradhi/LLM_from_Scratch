import torch
import torch.nn as nn
from typing import Optional
from model.attention import MultiHeadCausalSelfAttention
from model.feedforward import PositionwiseFeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Pre-norm architecture (LayerNorm BEFORE attention/FFN, not after).
        GPT-2 uses pre-norm. It's more stable to train than post-norm.

        Components:
          self.norm1 = nn.LayerNorm(d_model)
          self.attn  = MultiHeadCausalSelfAttention(d_model, n_heads, dropout)
          self.norm2 = nn.LayerNorm(d_model)
          self.ff    = PositionwiseFeedForward(d_model, d_ff, dropout)
          self.dropout = nn.Dropout(dropout)
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pre-norm residual connections:

        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

        WHY PRE-NORM: In post-norm (original paper), LayerNorm is after the
        residual. This means the residual stream is normalized, which can hurt
        gradient flow in deep networks. Pre-norm keeps the residual stream
        unnormalized — gradients flow more freely through the skip connection.
        """
        # Block 1: Attention with residual connection
        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask=key_padding_mask))
        
        # Block 2: Feed-Forward with residual connection
        x = x + self.dropout(self.ff(self.norm2(x)))
        
        return x