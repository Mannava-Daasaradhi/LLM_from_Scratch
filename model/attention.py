import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

def scaled_dot_product_attention(
    q: torch.Tensor,   # (B, H, T, d_k)
    k: torch.Tensor,   # (B, H, T, d_k)
    v: torch.Tensor,   # (B, H, T, d_k)
    mask: Optional[torch.Tensor] = None  # (B, 1, T, T) or (1, 1, T, T)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standalone function for scaled dot-product attention.
    Returns (output, attention_weights).
    output: (B, H, T, d_k)
    attention_weights: (B, H, T, T)
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask
        
    attention_weights = F.softmax(scores, dim=-1)
    output = attention_weights @ v
    
    return output, attention_weights

class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        d_model: total embedding dimension (must be divisible by n_heads)
        n_heads: number of attention heads
        d_k = d_model // n_heads: dimension per head

        Linear projections:
          W_q: d_model → d_model  (query projection)
          W_k: d_model → d_model  (key projection)
          W_v: d_model → d_model  (value projection)
          W_o: d_model → d_model  (output projection)

        Implement as a single combined QKV projection for efficiency:
          self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
          self.out_proj = nn.Linear(d_model, d_model, bias=False)

        Register causal mask as a buffer (not a parameter).
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask: upper triangular matrix of -inf
        # Shape: (1, 1, max_seq_len, max_seq_len)
        # mask[0, 0, i, j] = -inf if j > i else 0
        # Register as buffer so it moves to GPU with .to(device)
        max_seq = 5000
        mask = torch.triu(torch.ones(max_seq, max_seq), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        key_padding_mask: (batch_size, seq_len) — True where tokens are padding

        STEP BY STEP:
        1. Project to Q, K, V:
           qkv = self.qkv(x)                        # (B, T, 3*d_model)
           q, k, v = qkv.chunk(3, dim=-1)           # each (B, T, d_model)

        2. Reshape for multi-head:
           q = q.view(B, T, n_heads, d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
           k = k.view(B, T, n_heads, d_k).transpose(1, 2)
           v = v.view(B, T, n_heads, d_k).transpose(1, 2)

        3. Scaled dot-product attention:
           scores = q @ k.transpose(-2, -1) / sqrt(d_k)    # (B, n_heads, T, T)

        4. Apply causal mask:
           scores = scores + self.causal_mask[:, :, :T, :T]
           (adding -inf to future positions → softmax gives 0 weight to them)

        5. Apply key_padding_mask (if provided):
           If key_padding_mask[b, j] is True, set scores[b, :, :, j] = -inf
           This prevents attending to padding tokens.

        6. Softmax + dropout:
           attn_weights = F.softmax(scores, dim=-1)         # (B, n_heads, T, T)
           attn_weights = self.dropout(attn_weights)

        7. Weighted sum of values:
           out = attn_weights @ v                            # (B, n_heads, T, d_k)

        8. Reshape and project:
           out = out.transpose(1, 2).contiguous().view(B, T, d_model)
           out = self.out_proj(out)                          # (B, T, d_model)

        9. Return out

        IMPORTANT: Step 4 uses addition (not masking with bool tensor) because
        adding -inf before softmax is numerically equivalent to masking and is
        faster on GPU. Understand why: softmax(x + (-inf)) = 0.
        """
        B, T, C = x.size()

        # 1. Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape for multi-head
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Scaled dot-product attention
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        # 4. Apply causal mask
        scores = scores + self.causal_mask[:, :, :T, :T]

        # 5. Apply key_padding_mask
        if key_padding_mask is not None:
            mask_bool = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_bool, float('-inf'))

        # 6. Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. Weighted sum of values
        out = attn_weights @ v

        # 8. Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)

        # 9. Return out
        return out