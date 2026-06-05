import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Importing components we built in previous steps
from model.embedding import TokenEmbedding, RMSNorm
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
        rope_theta = getattr(config, 'rope_theta', 10000.0)

        # 1. Components
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)   # dropout on the token embeddings

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.dropout,
                max_seq_len=config.max_seq_len, rope_theta=rope_theta,
            )
            for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.d_model)   # final norm before projection
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 2. Weight Tying — share token embedding and LM-head weights.
        self.head.weight = self.token_emb.embedding.weight

        # 3. Parameter Initialization (GPT-2 convention)
        self.apply(self._init_weights)

        # Scaled init for the residual projections (attention out_proj + SwiGLU
        # down-projection w2): std *= 1/sqrt(2 * n_layers) so the residual stream
        # variance does not grow with depth.
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, T)
        targets: Optional[torch.Tensor] = None,   # (B, T) — if provided, compute loss
        key_padding_mask: Optional[torch.Tensor] = None,
        past_kvs: Optional[list] = None,          # per-layer (k, v) caches
        use_cache: bool = False,
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

        # 1. Token embeddings (+ dropout). Position is injected by RoPE inside
        #    attention, so there is no additive positional encoding here.
        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)

        # 2. Transformer Blocks
        if use_cache:
            # Incremental-decoding path: thread per-layer K/V caches through and
            # return them. Used only by generate(); training never sets use_cache.
            presents = []
            for i, block in enumerate(self.blocks):
                past = past_kvs[i] if past_kvs is not None else None
                x, present = block(x, key_padding_mask=key_padding_mask,
                                   past_kv=past, use_cache=True)
                presents.append(present)
            x = self.norm(x)
            logits = self.head(x)
            return logits, presents

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
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation loop with optional KV-caching.

        With use_cache=True (default) the prompt is processed once to prime per-layer
        K/V caches, then each new token is generated by running the model on just that
        single token — turning per-step cost from O(T) (recompute the whole prefix) into
        O(1). RoPE handles the absolute position via the cache offset. Output is identical
        to the cache-free path; use_cache=False forces the simple full-recompute loop.
        """
        self.eval()  # no dropout

        past_kvs = None
        for step in range(max_new_tokens):
            # 1. Choose the input for this step
            if use_cache and past_kvs is not None:
                idx_cond = input_ids[:, -1:]                      # only the newest token
            else:
                # First step (or no cache): feed the (truncated) prompt
                idx_cond = (input_ids if input_ids.size(1) <= self.config.max_seq_len
                            else input_ids[:, -self.config.max_seq_len:])

            # 2. Forward pass to get logits for the last position
            if use_cache:
                logits, past_kvs = self(idx_cond, past_kvs=past_kvs, use_cache=True)
            else:
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (1, vocab_size)

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