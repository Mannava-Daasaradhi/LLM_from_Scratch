import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from model.embedding import RotaryEmbedding, apply_rotary

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
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 max_seq_len: int = 5000, rope_theta: float = 10000.0):
        """
        Multi-head causal self-attention with rotary position embeddings (RoPE) and
        PyTorch's fused/flash scaled-dot-product-attention kernel.

        d_model: total embedding dimension (must be divisible by n_heads)
        n_heads: number of attention heads
        d_k = d_model // n_heads: dimension per head

        Single combined QKV projection for efficiency:
          self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
          self.out_proj = nn.Linear(d_model, d_model, bias=False)

        Causality is enforced by F.scaled_dot_product_attention(is_causal=True), so we
        no longer materialise a (max_seq, max_seq) mask buffer — that previously cost
        ~25M floats *per layer* in every checkpoint. The flash kernel applies the mask
        implicitly and runs in O(T) memory.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.d_k, max_seq_len=max_seq_len, theta=rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple] = None,
        use_cache: bool = False,
    ):
        """
        x: (batch_size, seq_len, d_model)
        key_padding_mask: (batch_size, seq_len) — True where tokens are padding
        past_kv: optional (k, v) cache from previous steps, each (B, H, T_past, d_k)
        use_cache: if True, also return the updated (k, v) cache for the next step

        1. Project to Q, K, V and split into heads → (B, n_heads, T, d_k)
        2. Apply RoPE to Q and K, offset by the number of cached positions so absolute
           positions stay correct during incremental decoding.
        3. Prepend any cached K/V, then run scaled dot-product attention (flash kernel
           on GPU). During training (no cache, no padding) this is the fast is_causal path.
        4. Re-merge heads and apply the output projection.

        Returns `out` normally, or `(out, (k, v))` when use_cache=True. The training /
        eval path never sets use_cache, so it is completely unchanged.
        """
        B, T, C = x.size()
        past_len = past_kv[0].size(2) if past_kv is not None else 0

        # 1. Project to Q, K, V and reshape for multi-head
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)   # (B, H, T, d_k)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 2. RoPE on Q and K, using absolute positions [past_len, past_len + T)
        cos, sin = self.rope(past_len + T, x.device)
        cos = cos[:, :, past_len:past_len + T, :]
        sin = sin[:, :, past_len:past_len + T, :]
        q, k = apply_rotary(q, k, cos, sin)

        # 3. Prepend cached keys/values (incremental decoding)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None

        dropout_p = self.dropout_p if self.training else 0.0
        if past_len == 0 and key_padding_mask is None:
            # Standard full-sequence path (training/eval and prompt prefill).
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        elif key_padding_mask is None and T == 1:
            # Single-token incremental decode: the lone query is the newest position,
            # so every cached key is in its past — no mask needed (fast kernel path).
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            # General case (multi-token with a cache, or padding present): build an
            # explicit additive mask. Query i has absolute position past_len+i and may
            # attend to key j iff j <= past_len + i (causal), plus padding.
            Tk = k.size(2)
            i_pos = torch.arange(T, device=x.device).unsqueeze(1) + past_len
            j_pos = torch.arange(Tk, device=x.device).unsqueeze(0)
            attn_mask = torch.where(j_pos <= i_pos, 0.0, float("-inf")).to(q.dtype)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, T, Tk).clone()
            if key_padding_mask is not None:
                pad = key_padding_mask.unsqueeze(1).unsqueeze(2)      # (B, 1, 1, Tk)
                attn_mask = attn_mask.masked_fill(pad, float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

        # 4. Re-merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        return (out, present) if use_cache else out