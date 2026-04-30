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
        Multiply output by sqrt(d_model) — this is the scaling from the original paper.
        Reason: keeps embedding magnitudes comparable to positional encodings.
        """
        return self.embedding(x) * math.sqrt(self.d_model)

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