import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Two linear layers with GELU activation between them.
        d_ff is typically 4 * d_model (2048 when d_model=512).

        Architecture:
          Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model) → Dropout

        Why GELU not ReLU: GELU is smoother and empirically works better for
        language models. GPT-2 uses GELU. ReLU has a hard zero that kills gradients.
        Understand this — you will be asked.

        Weight initialization: default PyTorch (Kaiming uniform) is fine here.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        Returns: (batch_size, seq_len, d_model)
        The FFN is applied independently to each position — no interaction
        between positions happens here. That's the attention's job.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x