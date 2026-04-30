
> **Domain:** AI/ML · **Year:** 1 · **Months:** 2–3 · **Daily Time Budget:** 2 hrs  
> **Legal Status:** ✅ Safe — no legal restrictions  
> **Stack:** PyTorch · CUDA · Python  
> **Signal:** GitHub stars + proves ML internals depth to any ML research/engineering interviewer

---

## 1. What This Project Is

You will build a **GPT-style transformer language model from scratch** — no HuggingFace, no nanoGPT copy-paste, no `transformers` library. Every component is written by you, understood by you, and tested by you.

The pipeline you will implement end-to-end:

```
Raw text (Shakespeare)
  → BPE Tokenizer (you implement the merge algorithm)
  → Token + Positional Embeddings
  → N × Transformer Decoder Blocks
      → Multi-Head Causal Self-Attention
      → Feed-Forward Network
      → LayerNorm + Residual connections
  → Linear projection → Softmax → Cross-entropy loss
  → AdamW optimizer with warmup + cosine decay
  → Generation: temperature / top-k / top-p sampling
```

By the end, running `python generate.py --prompt "To be or not"` produces coherent Shakespeare-style continuation. Validation perplexity is below 50.

This project is the **technical foundation for everything ML in this plan.** P6 (RAG) reuses your tokenizer. P11 (neural net in C) reuses your understanding of backprop. P18 (Circuit-Breaker paper) requires you to know transformer internals well enough to write about mechanistic interpretability. Build this right.

---

## 2. Problem Statement

Everyone who uses GPT-4 has an opinion about transformers. Almost no one can implement one from memory. The gap between "I've read Attention Is All You Need" and "I can implement scaled dot-product attention with a causal mask in numpy and explain every dimension" is enormous — and that gap is exactly what this project closes.

**Specific things you will be able to do after this project:**

- Derive the attention formula from scratch on a whiteboard
- Explain why we divide by √d_k and what happens if we don't
- Implement BPE tokenization without looking at any reference
- Explain why causal masking is necessary and implement it two different ways
- Debug a transformer that's not training (know which component to inspect first)
- Explain the difference between sinusoidal and learned positional encodings and when each is better
- Implement top-p (nucleus) sampling and explain why it's better than top-k for text quality

**Your definition of done:**

```bash
# Train on Shakespeare (~1MB text)
python train.py --config configs/shakespeare.yaml
# Prints: Epoch 10 | loss=1.23 | val_loss=1.41 | val_perplexity=4.09
# (perplexity < 50 is required; < 10 means you're doing well)

# Generate text
python generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt "To be or not to be" \
  --max_tokens 200 \
  --temperature 0.8 \
  --top_p 0.9

# Output: coherent Shakespeare-style text, not gibberish
# Example good output:
# "To be or not to be, that is the question of mine heart,
#  Whether 'tis nobler in the mind to suffer..."
```

---

## 3. Deliverables

| # | Deliverable | Where |
|---|-------------|-------|
| 1 | BPE tokenizer (train + encode + decode) | `tokenizer/bpe.py` |
| 2 | Full transformer model (all components) | `model/` |
| 3 | Training loop with AdamW + scheduler | `train.py` |
| 4 | Text generation script | `generate.py` |
| 5 | Config system (YAML) | `configs/shakespeare.yaml` |
| 6 | Test suite for each component | `tests/` |
| 7 | Training loss curve plot | `plots/training_curve.png` |
| 8 | README with architecture diagram + results table | root `README.md` |
| 9 | Gist: self-contained attention implementation (100 lines) | linked in README |

---

## 4. Repository Structure

```
llm-from-scratch/
├── tokenizer/
│   ├── __init__.py
│   ├── bpe.py            # BPE training + encode + decode
│   └── base.py           # BaseTokenizer abstract class
├── model/
│   ├── __init__.py
│   ├── attention.py      # MultiHeadCausalSelfAttention
│   ├── feedforward.py    # PositionwiseFeedForward
│   ├── embedding.py      # TokenEmbedding + PositionalEncoding
│   ├── block.py          # TransformerBlock (attention + FFN + norms + residuals)
│   └── transformer.py    # GPT: full model, forward, generate
├── configs/
│   └── shakespeare.yaml
├── data/
│   └── download.py       # downloads tinyshakespeare
├── tests/
│   ├── test_bpe.py
│   ├── test_attention.py
│   ├── test_transformer.py
│   └── test_generation.py
├── plots/
│   └── (generated during training)
├── checkpoints/          # .pt files saved here
├── train.py
├── generate.py
└── README.md
```

---

## 5. Step-by-Step Build Instructions

Build in this exact order. Each component is tested before moving to the next. Do not build the full model and then test — you will not be able to debug it.

---

### Step 1 — Data + Config (Session 1, ~30 min)

**Download the dataset:**

```bash
mkdir -p data && cd data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# 1.1MB, ~40K lines of Shakespeare
cd ..
```

**`data/download.py`** — write a script that:
1. Downloads the file if not present
2. Splits into train (90%) and val (10%) by character count
3. Saves `data/train.txt` and `data/val.txt`
4. Prints: `Train: 1,003,854 chars | Val: 111,540 chars`

**`configs/shakespeare.yaml`:**

```yaml
# Model architecture
model:
  vocab_size: 10000      # BPE vocabulary size
  d_model: 512           # embedding dimension
  n_heads: 8             # attention heads
  n_layers: 6            # transformer blocks
  d_ff: 2048             # feed-forward hidden dim (4 * d_model)
  max_seq_len: 256       # context window
  dropout: 0.1

# Training
training:
  batch_size: 64
  learning_rate: 3.0e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  warmup_steps: 100
  max_steps: 5000
  eval_interval: 500
  eval_steps: 100        # number of batches to average val loss over
  checkpoint_dir: checkpoints/
  log_interval: 50

# Data
data:
  train_file: data/train.txt
  val_file: data/val.txt
  tokenizer_path: tokenizer/shakespeare_bpe.json

# Device
device: cuda              # falls back to cpu if cuda unavailable
```

Write a `config.py` that loads this YAML into a dataclass:
```python
from dataclasses import dataclass
import yaml

@dataclass
class ModelConfig: ...
@dataclass
class TrainingConfig: ...
@dataclass
class Config: ...

def load_config(path: str) -> Config: ...
```

---

### Step 2 — BPE Tokenizer (Sessions 2–3, ~120 min)

**File:** `tokenizer/bpe.py`

This is the most underestimated component. Implement it fully before touching the model.

**What BPE does:**
BPE starts with a character-level vocabulary and repeatedly merges the most frequent pair of adjacent tokens into a new token, until the vocabulary reaches the target size. After training, it encodes text by greedily applying the learned merges in order.

**Full implementation spec:**

```python
import json
import re
from collections import defaultdict
from typing import Optional

class BPETokenizer:
    # Special tokens — always in vocabulary at fixed IDs
    PAD_TOKEN = "<pad>"   # ID 0
    UNK_TOKEN = "<unk>"   # ID 1
    BOS_TOKEN = "<bos>"   # ID 2
    EOS_TOKEN = "<eos>"   # ID 3

    def __init__(self):
        self.vocab: dict[str, int] = {}        # token string → ID
        self.id_to_token: dict[int, str] = {}  # ID → token string
        self.merges: list[tuple[str, str]] = [] # ordered merge rules
        self._trained = False

    def train(self, text: str, vocab_size: int = 10000) -> None:
        """
        Train BPE on text. After this call:
          - self.vocab has vocab_size entries
          - self.merges has (vocab_size - initial_vocab_size) entries in order

        ALGORITHM:
        1. Initialize character vocabulary:
           - Add 4 special tokens (IDs 0-3)
           - Split text into words (split on whitespace, keep punctuation)
           - Prepend 'Ġ' (U+0120) to each word to mark word boundaries
             (this is the GPT-2 convention — 'hello' becomes 'Ġhello')
           - Build initial vocab: every unique character in the corpus + special tokens
           - Represent each word as a tuple of characters: ('Ġ','h','e','l','l','o')

        2. Count word frequencies:
           word_freqs: dict mapping word_tuple → int count

        3. Repeat until vocab_size reached:
           a. Count all adjacent pair frequencies across all words
              (weighted by word frequency)
           b. Find the most frequent pair (break ties alphabetically)
           c. Merge that pair into a new token everywhere in word_freqs
           d. Add new token to vocab, record merge rule

        4. Store final vocab and merges.

        COMPLEXITY NOTE:
        Naive implementation is O(vocab_size * corpus_size). This is fine for
        1MB of text and 10K vocab. Do not optimize prematurely.
        """

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode text to token IDs using trained merge rules.

        ALGORITHM:
        1. Split text into words (same whitespace split as training)
        2. For each word:
           a. Start with character-level representation (with Ġ prefix)
           b. Apply merges in order (same order as training):
              - Scan word for the highest-priority merge
              - Apply it, repeat until no merges apply
        3. Convert tokens to IDs (use UNK_TOKEN ID for unknown tokens)
        4. If add_special_tokens: prepend BOS_TOKEN ID, append EOS_TOKEN ID
        5. Return list of int IDs

        IMPORTANT: Merges must be applied in training order, not greedily by
        frequency of the current text. This is what makes BPE deterministic.
        """

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        1. Convert IDs to token strings
        2. Skip special tokens if skip_special_tokens=True
        3. Join tokens, replacing 'Ġ' with ' '
        4. Strip leading/trailing whitespace
        """

    def save(self, path: str) -> None:
        """
        Save tokenizer to JSON file:
        {
          "vocab": {"token": id, ...},
          "merges": [["tok_a", "tok_b"], ...],
          "version": "1.0"
        }
        """

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from JSON file saved by save()."""

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
```

**Training the tokenizer — run this after implementing:**

```python
# In a script: tokenizer/train_tokenizer.py
from tokenizer.bpe import BPETokenizer

with open("data/train.txt") as f:
    text = f.read()

tok = BPETokenizer()
tok.train(text, vocab_size=10000)
tok.save("tokenizer/shakespeare_bpe.json")

# Sanity checks
test = "To be or not to be, that is the question."
encoded = tok.encode(test)
decoded = tok.decode(encoded)
assert decoded.strip() == test.strip(), f"Round-trip failed:\n  original: {test}\n  decoded:  {decoded}"
print(f"Vocab size: {tok.vocab_size}")
print(f"Encoded: {encoded[:10]}...")
print(f"Decoded: {decoded}")
print("Tokenizer training complete.")
```

**Tests in `tests/test_bpe.py`:**

1. `test_train_produces_correct_vocab_size` — train on small text, assert `tok.vocab_size == target`
2. `test_encode_decode_roundtrip` — encode then decode, assert text recovered (ignoring whitespace normalization)
3. `test_special_tokens_in_vocab` — assert PAD/UNK/BOS/EOS are IDs 0/1/2/3
4. `test_bos_eos_added` — `encode("hello", add_special_tokens=True)` starts with BOS ID and ends with EOS ID
5. `test_unknown_token_maps_to_unk` — encode character not in vocab, assert UNK ID appears
6. `test_save_load_produces_identical_tokenizer` — train, save, load, encode same text, assert identical IDs
7. `test_merges_applied_in_order` — manually set 2 merge rules, encode short text, verify correct merge applied
8. `test_word_boundary_marker` — 'hello world' encodes so that 'world' token starts with Ġ

---

### Step 3 — Embeddings (Session 4, ~45 min)

**File:** `model/embedding.py`

```python
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        Add PE for positions 0..seq_len-1.
        self.pe[:, :seq_len, :] is shape (1, seq_len, d_model) — broadcasts over batch.
        Apply dropout after addition.
        Returns: (batch_size, seq_len, d_model)
        """


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        """
        Alternative to sinusoidal: just nn.Embedding(max_seq_len, d_model).
        During forward, create position indices [0, 1, ..., seq_len-1] and embed them.
        Add to token embeddings. Apply dropout.
        GPT-2 uses this. Slightly better in practice for fixed-length contexts.
        Implement both — use sinusoidal by default, make it a config option.
        """
```

**Tests in `tests/test_embedding.py`** (add to test_transformer.py):

1. `test_token_embedding_shape` — input (2, 10), output (2, 10, 512)
2. `test_sinusoidal_pe_shape` — same shape check
3. `test_sinusoidal_pe_is_not_learned` — assert PE buffer not in `model.parameters()`
4. `test_pe_values_sin_cos` — check `pe[0, 0, 0]` == sin(0) == 0.0 and `pe[0, 0, 1]` == cos(0) == 1.0
5. `test_learned_pe_is_learned` — assert learned PE IS in `model.parameters()`

---

### Step 4 — Multi-Head Causal Self-Attention (Sessions 5–6, ~120 min)

**File:** `model/attention.py`

This is the core of the transformer. Take your time. Understand every line.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
```

**Tests in `tests/test_attention.py`:**

1. `test_output_shape` — input (2, 10, 512), output (2, 10, 512)
2. `test_causal_mask_prevents_future_attention` — set sequence [A, B, C]. Assert attention weight from position 0 to positions 1, 2 is 0 (not just small — exactly 0 after softmax of -inf)
3. `test_different_heads_different_weights` — extract attention weights per head, assert they are not all identical
4. `test_single_token_works` — seq_len=1, no errors
5. `test_gradient_flows` — forward pass, compute `.sum().backward()`, assert `qkv.weight.grad is not None`
6. `test_attention_weights_sum_to_one` — for each (batch, head, query_pos), attention weights over keys sum to 1.0 (within 1e-5)

**Implement a standalone `scaled_dot_product_attention` function** (not a class — just the math):

```python
def scaled_dot_product_attention(
    q: torch.Tensor,   # (B, H, T, d_k)
    k: torch.Tensor,   # (B, H, T, d_k)
    v: torch.Tensor,   # (B, H, T, d_k)
    mask: torch.Tensor = None  # (B, 1, T, T) or (1, 1, T, T)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (output, attention_weights).
    output: (B, H, T, d_k)
    attention_weights: (B, H, T, T)
    """
```

This is separate from the class so you can unit test the math in isolation. Post this function as your Gist — it should be the cleanest 30 lines you've ever written.

---

### Step 5 — Feed-Forward Network (Session 7, ~30 min)

**File:** `model/feedforward.py`

```python
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        Returns: (batch_size, seq_len, d_model)
        The FFN is applied independently to each position — no interaction
        between positions happens here. That's the attention's job.
        """
```

Test: shape in == shape out. Gradient flows. No more tests needed for this component — it's 5 lines.

---

### Step 6 — Transformer Block (Session 7 continued, ~30 min)

**File:** `model/block.py`

```python
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

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
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
```

Test: shape preserved. Residual means output ≠ zeroed input (even with dropout in eval mode).

---

### Step 7 — Full GPT Model (Session 8, ~60 min)

**File:** `model/transformer.py`

```python
class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
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

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, T)
        targets: torch.Tensor = None,   # (B, T) — if provided, compute loss
        key_padding_mask: torch.Tensor = None
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

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,      # (1, T) — prompt tokens
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation loop.

        For each step:
        1. If input_ids longer than max_seq_len: truncate to last max_seq_len tokens
        2. Forward pass → logits (1, T, vocab_size) → take last position: (1, vocab_size)
        3. Apply temperature: logits = logits / temperature
           (temperature < 1: sharper/more deterministic; > 1: flatter/more random)

        4. Apply top-k filtering (if top_k is not None):
           Keep only the top_k highest logit values.
           Set all others to -inf.
           top_k_values, _ = torch.topk(logits, top_k)
           threshold = top_k_values[:, -1].unsqueeze(-1)
           logits = logits.masked_fill(logits < threshold, float('-inf'))

        5. Apply top-p (nucleus) filtering (if top_p is not None):
           Sort logits descending. Compute cumulative softmax probabilities.
           Find cutoff where cumulative prob exceeds top_p.
           Set all tokens beyond cutoff to -inf.
           (This dynamically selects the smallest set of tokens whose combined
            probability exceeds top_p — better than top_k because it adapts
            to the probability distribution shape.)

        6. Sample from distribution:
           probs = F.softmax(logits, dim=-1)
           next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        7. Append to sequence:
           input_ids = torch.cat([input_ids, next_token], dim=1)

        8. Stop if EOS token sampled.

        9. Return full generated sequence tensor.

        IMPORTANT: top_k and top_p can be used together. Apply top_k first,
        then top_p on the remaining candidates.
        """

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
```

**Tests in `tests/test_transformer.py`:**

1. `test_forward_shape` — input (2, 32), logits shape (2, 32, vocab_size)
2. `test_forward_with_targets_returns_loss` — targets provided, loss is a scalar tensor
3. `test_loss_decreases_on_single_batch` — run 10 gradient steps on one batch, assert loss[9] < loss[0]
4. `test_generate_returns_longer_sequence` — `generate` with max_new_tokens=20 returns tensor with seq_len > input
5. `test_generate_respects_temperature` — same prompt, temp=0.01 vs temp=2.0, assert outputs differ
6. `test_parameter_count` — with d_model=64, n_heads=4, n_layers=2, vocab=1000: assert param count matches manual calculation
7. `test_weight_tying` — assert `model.token_emb.embedding.weight.data_ptr() == model.head.weight.data_ptr()`

---

### Step 8 — Training Loop (Sessions 9–10, ~120 min)

**File:** `train.py`

```python
"""
Full training script. Run as:
  python train.py --config configs/shakespeare.yaml
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml, argparse, math, time, os
from pathlib import Path
from miniflow import ExperimentTracker   # use your P1a tracker!

# Dataset
class TextDataset(Dataset):
    """
    Loads tokenized text and returns (input_ids, targets) pairs.
    input_ids:  tokens[i : i + seq_len]
    targets:    tokens[i+1 : i + seq_len + 1]  (next-token prediction)
    Stride: seq_len // 2 (50% overlap between consecutive windows — more training examples)
    """
    def __init__(self, filepath: str, tokenizer, seq_len: int):
        with open(filepath) as f:
            text = f.read()
        self.tokens = torch.tensor(tokenizer.encode(text, add_special_tokens=False))
        self.seq_len = seq_len
        self.stride = seq_len // 2

    def __len__(self):
        return (len(self.tokens) - self.seq_len - 1) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y
```

**Optimizer: AdamW with correct weight decay:**
```python
def configure_optimizer(model, config):
    """
    Apply weight decay only to weight matrices, NOT to:
    - Bias terms
    - LayerNorm weights and biases
    - Embedding weights

    Reason: decaying embeddings and LayerNorm parameters hurts performance.

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=config.learning_rate,
                              betas=(config.beta1, config.beta2))
    """
```

**Learning rate scheduler — cosine with warmup:**
```python
def get_lr(step: int, config) -> float:
    """
    Linear warmup for warmup_steps, then cosine decay to 10% of max LR.

    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    # cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))

    min_lr = max_lr * 0.1
    """
```

**Main training loop:**
```python
def train(config):
    # Setup
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    tracker = ExperimentTracker("llm_shakespeare")
    tracker.log_params({
        "d_model": config.model.d_model,
        "n_layers": config.model.n_layers,
        "n_heads": config.model.n_heads,
        "vocab_size": config.model.vocab_size,
        "batch_size": config.training.batch_size,
        "max_steps": config.training.max_steps,
    })

    # Load tokenizer
    # Build datasets + dataloaders
    # Build model → move to device
    # Log param count
    # Configure optimizer
    # Training loop:
    for step in range(max_steps):
        # Set LR for this step
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward
        model.train()
        x, y = next(train_iter)   # use itertools.cycle on dataloader
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()

        # Gradient clipping — critical for transformer stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        optimizer.step()

        # Logging
        if step % config.training.log_interval == 0:
            tracker.log_metric("train_loss", loss.item(), step=step)
            tracker.log_metric("lr", lr, step=step)
            print(f"Step {step:5d} | loss={loss.item():.4f} | lr={lr:.2e}")

        # Evaluation
        if step % config.training.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, config.training.eval_steps)
            val_ppl = math.exp(val_loss)
            tracker.log_metric("val_loss", val_loss, step=step)
            tracker.log_metric("val_perplexity", val_ppl, step=step)
            print(f"  VAL | loss={val_loss:.4f} | perplexity={val_ppl:.2f}")

            # Save checkpoint if best so far
            save_checkpoint(model, optimizer, step, val_loss, config)

    tracker.finish()

@torch.no_grad()
def evaluate(model, val_loader, device, n_steps) -> float:
    """Average val loss over n_steps batches."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= n_steps: break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    return sum(losses) / len(losses)

def save_checkpoint(model, optimizer, step, val_loss, config):
    """Save to checkpoints/step_{step}.pt. Keep only best (lowest val_loss)."""
```

---

### Step 9 — Generation Script (Session 10 continued, ~30 min)

**File:** `generate.py`

```python
"""
python generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt "To be or not to be" \
  --max_tokens 200 \
  --temperature 0.8 \
  --top_p 0.9
"""
import torch, argparse
from tokenizer.bpe import BPETokenizer
from model.transformer import GPT
from config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    tokenizer = BPETokenizer.load(config.data.tokenizer_path)
    model = GPT(config.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(args.prompt)]).long()

    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Decode and print (only the new tokens)
    new_tokens = output_ids[0, input_ids.shape[1]:].tolist()
    generated_text = tokenizer.decode(new_tokens)
    print(f"\n=== PROMPT ===\n{args.prompt}")
    print(f"\n=== GENERATED ===\n{generated_text}")

if __name__ == "__main__":
    main()
```

---

### Step 10 — Training + Results (Sessions 11–12, ~180 min)

Actually train the model. This is not optional.

```bash
# Step 1: Train BPE tokenizer
python tokenizer/train_tokenizer.py
# Expected: Vocab size: 10000. Round-trip test passes.

# Step 2: Train model
python train.py --config configs/shakespeare.yaml
# Watch loss decrease. Should take 30-90 minutes depending on GPU.
# Target: val_perplexity < 50 (achievable; < 10 with GPU and patience)

# Step 3: Generate
python generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt "To be or not to be" \
  --max_tokens 200 \
  --temperature 0.8 \
  --top_p 0.9
```

**What to do if loss is not decreasing:**
- Check that causal mask is applied (without it, the model leaks future tokens)
- Check weight tying is set up correctly
- Check learning rate: if loss explodes, LR is too high; if it barely moves, LR too low
- Check gradient clipping is happening (log `grad_norm` to verify)
- Try reducing batch size if CUDA OOM

**Plot training curves:**
```python
# plots/plot_curves.py
# Pull loss from MiniFlow ExperimentTracker (P1a!)
# Plot train_loss and val_loss on same axes
# Save to plots/training_curve.png
```

---

## 6. Definition of Done

```bash
# 1. Tokenizer round-trip
python -c "
from tokenizer.bpe import BPETokenizer
tok = BPETokenizer.load('tokenizer/shakespeare_bpe.json')
text = 'To be or not to be, that is the question.'
assert tok.decode(tok.encode(text)).strip() == text.strip()
print('Tokenizer OK')
"

# 2. Model forward pass
python -c "
import torch
from model.transformer import GPT
from config import ModelConfig
cfg = ModelConfig(vocab_size=1000, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=32, dropout=0.0)
model = GPT(cfg)
x = torch.randint(0, 1000, (2, 16))
logits, _ = model(x)
assert logits.shape == (2, 16, 1000)
print('Model forward OK')
"

# 3. Loss decreases
python -c "
# Run 50 steps on a single batch, assert loss[49] < loss[0]
# (implement this inline or call your training loop with max_steps=50)
print('Training loop OK')
"

# 4. Generation produces text
python generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt 'To be or not' \
  --max_tokens 100
# Must output something that looks like English (not random symbols)

# 5. Validation perplexity
# Inspect training logs: val_perplexity must be < 50 at any point during training

# 6. Tests pass
pytest tests/ -v
# 0 failed

# 7. GitHub
# - All code pushed
# - Training curve PNG in plots/ committed
# - README shows: architecture, training config, sample generated text, perplexity result
# - Gist linked: standalone scaled_dot_product_attention implementation
```

---

## 7. Things You Must Be Able to Explain After This Project

After completing P2, you must be able to answer these questions without notes. Practice out loud.

1. Why do we scale attention scores by 1/√d_k? What happens if we don't?
2. What is the causal mask and why is it necessary for language modeling?
3. What is weight tying? Why does it work?
4. What is the difference between top-k and top-p sampling? When is each better?
5. Why is pre-norm more stable than post-norm for deep transformers?
6. Why does the FFN use GELU instead of ReLU?
7. What does BPE stand for? Walk me through the algorithm step by step.
8. What is perplexity? If perplexity is 20, what does that mean intuitively?
9. Why do we apply weight decay only to weight matrices and not biases/LayerNorm?
10. What is gradient clipping and why do transformers need it more than CNNs?

---

## 8. Common Mistakes to Avoid

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Forgetting the causal mask | Model sees future tokens during training, learns trivially, generates garbage | Test mask explicitly — assert future positions have 0 attention weight |
| Post-norm instead of pre-norm | Unstable training at 6+ layers | LayerNorm goes BEFORE attention/FFN, not after |
| Not tying weights | 5M extra parameters, slightly worse perplexity | `self.head.weight = self.token_emb.embedding.weight` |
| Applying weight decay to all params | Decays LayerNorm/bias → bad performance | Separate param groups in optimizer |
| Temperature = 0 in generate() | Division by zero | Handle `temperature <= 0` — either raise ValueError or set to 1e-8 |
| Top-p applied before top-k | Can produce unexpected results | Always apply top-k first, then top-p |
| Not clipping gradients | Loss explodes mid-training | `clip_grad_norm_` after backward, before optimizer step — always |
| Using DataLoader without `pin_memory=True` | Slow CPU→GPU transfer | `DataLoader(..., pin_memory=True, num_workers=4)` |
| Calling `model.train()` after eval | Forgets to switch back | Set `model.train()` at the start of every training step |

---

## 9. How This Connects to the Rest of the Plan

| Future Project | How P2 Feeds It |
|----------------|----------------|
| **P6** (RAG pipeline) | Your BPE tokenizer and embedding layer are reused directly |
| **P11** (Neural net in C) | You implement backprop here in PyTorch; P11 does it in raw C |
| **P18** (Circuit-Breaker paper) | You need transformer internals to understand circuits and features |
| **P39** (LLM Inference Server) | This is the model you'll eventually serve at scale |
| **P49** (Distributed Training) | You'll parallelize training of a model like this one |
| **P50** (Autonomous Agent Security) | You need to understand the model to red-team it |
| **All ML projects** | Attention mechanism understanding is assumed from here on |

---

## 10. Time Breakdown

| Session | What You Do | Time |
|---------|-------------|------|
| 1 | Data download + config system | 30 min |
| 2–3 | BPE tokenizer + tests + train tokenizer | 120 min |
| 4 | Embeddings (token + sinusoidal + learned) | 45 min |
| 5–6 | Multi-head causal self-attention + tests | 120 min |
| 7 | FFN + TransformerBlock + tests | 60 min |
| 8 | Full GPT model + tests | 60 min |
| 9–10 | Training loop + generation script | 120 min |
| 11–12 | Actually train + plot curves + polish | 180 min |
| **Total** | | **~12 hrs across Months 2–3** |

At 2 hrs/day this is a 6-day focused sprint. Do it in one week — context switching kills transformer debugging.

---


