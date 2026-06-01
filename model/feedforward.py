import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network (Shazeer, 2020), the FFN used by LLaMA/PaLM/Mistral.

    A plain FFN is `Linear -> activation -> Linear`. SwiGLU replaces the single input
    projection with a *gated* one: it computes two projections of x, passes one through
    SiLU, and multiplies them elementwise before the down-projection:

        FFN(x) = W2( SiLU(W1 x) * (W3 x) )

    The multiplicative gate lets the network modulate which features pass through,
    which empirically beats GELU/ReLU FFNs at equal compute. Because there are three
    matrices instead of two, d_ff is commonly set to ~2/3 of the GELU width to keep
    the parameter count comparable. No biases (LLaMA convention).
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate branch
        self.w3 = nn.Linear(d_model, d_ff, bias=False)   # value branch
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # down projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        Applied independently to each position; no cross-position interaction here
        (that is attention's job).
        """
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.dropout(x)


# Backwards-compatible alias: older code / configs referred to the FFN as
# PositionwiseFeedForward. It now maps to the SwiGLU implementation.
PositionwiseFeedForward = SwiGLU
