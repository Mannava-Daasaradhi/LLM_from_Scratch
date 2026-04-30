import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Importing components we built in previous steps
from model.embedding import TokenEmbedding, SinusoidalPositionalEncoding
from model.block import TransformerBlock

# We assume config.py exists based on Step 1 of the master plan
# If you haven't made it yet, here is a quick dummy structure so the code runs:
# class ModelConfig: vocab_size=10000; d_model=512; n_heads=8; n_layers=6; d_ff=2048; max_seq_len=256; dropout=0.1

class GPT(nn.Module):
    def __init__(self, config):
        """
        Components:
          self.token_emb = TokenEmbedding(vocab_size, d_model)
          self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
          self.blocks    = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])
          self.norm      = nn.LayerNorm(d_model)   # final norm before projection
          self.head      = nn.Linear(d_model, vocab_size, bias=False)

        WEIGHT TYING: share weights between token_emb.embedding.weight and head.weight.
        self.head.weight = self.token_emb.embedding.weight
        Reason: the input and output vocabularies are the same. Tying reduces
        parameters by vocab_size * d_model (~5M params for our config) and
        empirically improves performance.

        Parameter initialization:
          - All Linear layers: N(0, 0.02) — GPT-2 convention
          - All LayerNorm: weight=1, bias=0
          - Apply special scaling to residual projections (out_proj and second
            FFN linear): multiply std by 1/sqrt(2 * n_layers)
            Reason: prevents residual stream from growing with depth.
        """
        super().__init__()
        self.config = config
        
        # 1. Components
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 2. Weight Tying
        self.head.weight = self.token_emb.embedding.weight
        
        # 3. Parameter Initialization
        self.apply(self._init_weights)
        
        # Apply special scaled initialization to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('linear2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, T)
        targets: Optional[torch.Tensor] = None,   # (B, T) — if provided, compute loss
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        1. x = self.token_emb(input_ids)    # (B, T, d_model)
        2. x = self.pos_enc(x)              # add positional encoding
        3. for block in self.blocks:
               x = block(x, key_padding_mask)
        4. x = self.norm(x)                 # final LayerNorm
        5. logits = self.head(x)            # (B, T, vocab_size)

        6. If targets provided:
               loss = F.cross_entropy(
                   logits.view(-1, vocab_size),
                   targets.view(-1),
                   ignore_index=PAD_TOKEN_ID   # don't compute loss on padding
               )
               return logits, loss
           Else:
               return logits, None
        """
        B, T = input_ids.size()
        
        # 1 & 2. Embeddings + Positional Encoding
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)
        
        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
            
        # 4 & 5. Final Norm & Head
        x = self.norm(x)
        logits = self.head(x)
        
        # 6. Loss Calculation
        loss = None
        if targets is not None:
            PAD_TOKEN_ID = 0  # Assuming 0 based on our BPE tokenizer
            # NOTE: label_smoothing is intentionally NOT applied here.
            # This path is used by evaluate() during validation — val loss must be
            # clean cross-entropy so it is comparable across runs and correctly
            # signals overfitting.  Label smoothing is applied only in the training
            # loop in train.py where it acts as a regulariser on training loss only.
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=PAD_TOKEN_ID,
            )
            
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,      # (1, T) — prompt tokens
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation loop.
        ... (Documentation preserved from prompt) ...
        """
        self.eval() # Ensure we are in eval mode (no dropout)
        
        for _ in range(max_new_tokens):
            # 1. Truncate context if it gets too long
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            
            # 2. Forward pass to get logits for the last token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # (1, vocab_size)
            
            # 3. Apply temperature
            if temperature <= 0.0:
                # Greedy decoding for T=0
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                
                # 4. Top-k filtering
                if top_k is not None:
                    top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    threshold = top_k_values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < threshold, float('-inf'))
                    
                # 5. Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter back to the original ordering
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))
                    
                # 6. Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            # 7. Append to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 8. Stop early if EOS token is generated (ID 3 based on our BPE)
            if next_token.item() == 3:
                break
                
        # 9. Return full generated sequence
        return input_ids

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        # Using a set ensures we don't double count tied weights
        n_params = sum(p.numel() for p in set(self.parameters()) if p.requires_grad)
        return n_params