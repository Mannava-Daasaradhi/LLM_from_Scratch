import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        Standard nn.Embedding layer.
        Weight initialization: N(0, 1/sqrt(d_model))
        This scaling is important — prevents embeddings from dominating
        the positional signal early in training.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        nn.init.normal_(self.embedding.weight, mean=0, std=1/math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len) — token IDs
        Returns: (batch_size, seq_len, d_model)

        NOTE: no sqrt(d_model) scaling. That scaling (from the original Transformer)
        balanced token embeddings against *additive* positional encodings. We now use
        RoPE (rotary), which is applied inside attention and adds nothing to the
        residual stream, so the GPT-2 convention of unscaled 0.02-init embeddings
        (also tied to the LM head) is the correct choice here.
        """
        return self.embedding(x)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        """
        Implements the sinusoidal PE from 'Attention Is All You Need':
          PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Precompute PE matrix of shape (1, max_seq_len, d_model) and register
        as a buffer (not a parameter — it's not learned).

        IMPLEMENTATION STEPS:
        1. Create position tensor: shape (max_seq_len, 1) — values 0..max_seq_len-1
        2. Create div_term: shape (d_model//2,)
           div_term = exp(arange(0, d_model, 2) * (-log(10000.0) / d_model))
           (this is the numerically stable way to compute 1/10000^(2i/d_model))
        3. pe = zeros(max_seq_len, d_model)
        4. pe[:, 0::2] = sin(position * div_term)
        5. pe[:, 1::2] = cos(position * div_term)
        6. pe = pe.unsqueeze(0)  → shape (1, max_seq_len, d_model)
        7. self.register_buffer('pe', pe)
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Implement above
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        Add PE for positions 0..seq_len-1.
        self.pe[:, :seq_len, :] is shape (1, seq_len, d_model) — broadcasts over batch.
        Apply dropout after addition.
        Returns: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        """
        Alternative to sinusoidal: just nn.Embedding(max_seq_len, d_model).
        During forward, create position indices [0, 1, ..., seq_len-1] and embed them.
        Add to token embeddings. Apply dropout.
        GPT-2 uses this. Slightly better in practice for fixed-length contexts.
        Implement both — use sinusoidal by default, make it a config option.
        """
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        x = x + self.embedding(positions).unsqueeze(0)
        return self.dropout(x)


class RMSNorm(nn.Module):
    """
    Root-Mean-Square LayerNorm (Zhang & Sennrich, 2019), as used by LLaMA/Mistral.

    Drops the mean-centering and bias of LayerNorm — it only rescales by the RMS of
    the activations and applies a learned per-channel gain. Cheaper and empirically
    as good or better for transformers. Computed in fp32 for numerical stability
    even under bf16/fp16 autocast.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight.float() * xf).to(in_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by splitting it in half: [a, b] -> [-b, a]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    q, k:     (B, H, T, head_dim)
    cos, sin: (1, 1, T, head_dim)
    """
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (Su et al., 2021 — RoPE).

    Instead of *adding* a positional vector to the token embedding, RoPE rotates the
    query/key vectors by an angle proportional to their absolute position. The dot
    product in attention then depends only on *relative* position, which generalises
    better to long contexts and adds zero learned parameters / nothing to the
    residual stream. cos/sin tables are cached lazily and never stored in the
    checkpoint (persistent=False).
    """
    def __init__(self, head_dim: int, max_seq_len: int = 5000, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head dimension"
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._cached_len = 0

    def _build_cache(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))   # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)             # (T, head_dim)
        self._cos_cached = emb.cos()[None, None, :, :]      # (1, 1, T, head_dim)
        self._sin_cached = emb.sin()[None, None, :, :]
        self._cached_len = seq_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if (self._cos_cached is None or seq_len > self._cached_len
                or self._cos_cached.device != device):
            self._build_cache(max(seq_len, self.max_seq_len), device)
        return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]