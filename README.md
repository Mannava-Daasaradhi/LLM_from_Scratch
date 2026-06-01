# LLM from Scratch 🧠

A GPT-style language model built entirely from scratch in PyTorch. Every component is
hand-written — a BPE tokenizer, multi-head causal self-attention, transformer blocks, and
a full training loop — using a modern LLaMA-style recipe: **RoPE** rotary positions,
**RMSNorm**, **SwiGLU** feed-forward, and **flash attention**. It trains on a ~24 MB
public-domain corpus of 25 classic books assembled from Project Gutenberg.

---

## Project Structure

```
LLM_from_Scratch/
├── config.py                  # Typed config dataclasses + YAML loader
├── configs/
│   └── shakespeare.yaml       # All hyperparameters in one place
├── data/
│   ├── download.py            # Assemble the corpus (Gutenberg classics / Shakespeare)
│   ├── input.txt              # Raw concatenated corpus
│   ├── train.txt / val.txt    # 90/10 split (+ cached *.bin token streams)
├── model/
│   ├── attention.py           # Multi-head causal self-attention (RoPE + flash SDPA)
│   ├── block.py               # Transformer block (pre-norm RMSNorm + residuals)
│   ├── embedding.py           # Token embedding, RMSNorm, RoPE (+ legacy PE classes)
│   ├── feedforward.py         # SwiGLU feed-forward network
│   └── transformer.py         # Full GPT model + generation
├── tokenizer/
│   ├── bpe.py                 # Byte-Pair Encoding tokenizer (fast rank-based encode)
│   ├── train_tokenizer.py     # Train a BPE vocab from text (CLI)
│   └── base.py                # Abstract base class
├── plots/
│   └── plot_curves.py         # Plot train/val loss curves from logs
├── tests/                     # Unit tests for each component
├── check_overfit.py           # Overfitting / architecture diagnostic
├── train.py                   # Main training loop
├── generate.py                # Text generation script
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the corpus

```bash
python data/download.py                       # 'classics': 25 Gutenberg books (~24 MB)
# alternatives:
python data/download.py --dataset shakespeare # complete works of Shakespeare only (~5 MB)
python data/download.py --dataset tinyshakespeare
```

This downloads, strips the Gutenberg boilerplate, concatenates, and writes a 90/10
`train.txt` / `val.txt` split.

### 3. Train the BPE tokenizer

```bash
python -m tokenizer.train_tokenizer \
    --input data/train.txt \
    --vocab-size 10000 \
    --output tokenizer/bpe.json \
    --max-chars 3000000        # learn merges from a 3 MB sample (generalises to the full corpus)
```

### 4. Train the model

```bash
python train.py --config configs/shakespeare.yaml
```

Training logs print every `log_interval` steps. Validation (loss, perplexity, **next-token
accuracy**) runs every `eval_interval` steps. The best checkpoint is saved to
`checkpoints/best.pt` — a lean file containing only model weights + config (no optimizer
state, no attention-mask buffers).

### 5. Generate text

```bash
python generate.py \
    --checkpoint checkpoints/best.pt \
    --prompt "To be or not to be" \
    --max_tokens 200 \
    --temperature 0.8 \
    --top_k 40
```

---

## Model Architecture

A decoder-only transformer (GPT-style) with a modern component set:

| Component | Choice | Why |
|---|---|---|
| Positional encoding | **RoPE** (rotary) | Relative-position aware, zero params, nothing added to the residual stream |
| Attention | Multi-head causal self-attention via **flash SDPA** | `F.scaled_dot_product_attention(is_causal=True)` — fast, O(T) memory, no mask buffer |
| Normalisation | **RMSNorm** (pre-norm) | Cheaper than LayerNorm, no bias/mean-centering, stable gradient flow |
| Activation | **SwiGLU** | Gated FFN; beats GELU/ReLU MLPs at equal compute |
| Weight tying | Token emb ↔ LM head | Saves ~5M params; improves generalisation |
| Tokenizer | **Byte-level BPE** + GPT-2-style regex pre-tokenizer | Preserves whitespace/newlines; exact round-trip; no true unknowns |
| Init | N(0, 0.02) + residual scaling | GPT-2 convention; controls residual stream growth with depth |
| Precision | **bf16 autocast** | ~2× faster on modern GPUs, no GradScaler needed |
| Generation | **KV-caching** (optional) | Reuses cached keys/values so each token costs O(1) compute instead of O(T) recompute (see note below) |

Default hyperparameters (`configs/shakespeare.yaml`):

```
d_model:     512    n_heads:  8    n_layers: 6
d_ff:       1536    dropout: 0.1   max_seq_len: 512
vocab_size: 10000   rope_theta: 10000
~25.6M parameters
```

---

## Training Details

### Optimiser
Fused AdamW with selective weight decay — weight matrices decay; biases, norms, and
embeddings do not.

### Learning Rate Schedule
Linear warmup for `warmup_steps`, then cosine decay to 10% of peak LR.

### Data Pipeline
The corpus is tokenised **once** and cached to `data/*.txt.bin` (uint16). Each training
step samples `batch_size` random windows from the token stream (nanoGPT-style) rather than
iterating fixed windows — endless varied batches and far better data utilisation.

### Efficiency
- **Flash attention** via `F.scaled_dot_product_attention` (replaces a manual softmax and
  a 5000×5000 mask buffer that previously bloated every checkpoint to ~888 MB).
- **bf16 autocast** + TF32 matmuls on CUDA; **fused AdamW**.
- Optional `torch.compile` (`compile: true`) and gradient accumulation (`grad_accum_steps`).

### Regularisation
With a ~24 MB corpus the model is far better matched to the data, so heavy regularisation
is no longer needed:

- **Dropout** `0.1` (down from `0.4`) — applied to embeddings, attention weights, and FFN.
- **Weight decay** `0.1` on weight matrices only.
- **Label smoothing** `0.0` (off — with enough data it mostly just slows learning).
- **Early stopping** `patience: 10` eval intervals — a safety net, rarely needed now.

> The earlier version used the *opposite* strategy (38M params on ~0.3M tokens, fought with
> dropout 0.4 + weight decay + label smoothing + aggressive early stopping). That model
> overfit instantly — its best validation loss came at the very first eval (step 500) and
> only got worse. The fix was to match data to capacity, not to crank up regularisation.

### Next-Token Accuracy
Validation reports top-1 next-token accuracy (the fraction of positions where `argmax`
of the logits equals the true next token), alongside loss and perplexity.

---

## Results

Trained on the 24 MB classics corpus (10K **byte-level** BPE vocab, ~25.6M params,
512-token context, RTX 4090, bf16). Validation reports clean cross-entropy loss,
perplexity, and top-1 next-token accuracy.

| | Old model | **This model** |
|---|---|---|
| Best val loss | 6.67 (at step 500, then **diverged**) | **3.72** (step 2000) |
| Best val perplexity | 786 | **41.1** |
| Next-token accuracy | — | **~33%** |
| Behaviour | overfit instantly; best at first eval | smooth descent, then early-stopped on overfit |
| Generated structure | run-on, no line breaks | **verse lines, stanzas, speaker labels** |
| Context window | 256 | 512 |
| Checkpoint size | 888 MB | 102 MB |
| Full-corpus tokenise | timed out (O(words×merges)) | ~5 s |

Validation perplexity over training (every 500 steps): `53 → 44 → 42 → 41 → 42 → 44 …`
— a smooth descent to a best of ~41 around step 2000, after which val loss rises as the
~25.6M-param model overfits the corpus. **Early stopping (patience 10) fired at step 7000**
and the best checkpoint (step 2000) was kept — the anti-overfit safety net working exactly
as intended, in stark contrast to the old model that diverged from step 500.

> **Comparing perplexity across tokenizers:** the byte-level tokenizer emits tokens for
> whitespace too, which are highly predictable, so its per-token perplexity isn't directly
> comparable to a whitespace-stripped tokenizer. The clean signals are that next-token
> accuracy rose (≈25% → ≈33%) *and* the model now reproduces document structure.

Sample (`--prompt "To be or not to be" --temperature 0.8 --top_k 40`):

```
Pursued on his mistress' ear to make
The man in my closet? To be sure, a very vile fellow

SEBASTIAN.
You'll get the fool here,
And get you gone.

GONZALO.
There's no man in all this, unless thou art not a soldier.
```

The output is grammatical English with **proper verse line breaks, stanza spacing, and
speaker labels** (and real character names) — the byte-level regex pre-tokenizer preserves
whitespace and gives an exact encode→decode round-trip for any text.

**Remaining limitations:**
- The small model still repeats speaker labels and lacks long-range plot coherence.
- Occasional `�` appears when sampling lands mid-way through a multi-byte UTF-8 character
  (the Gutenberg corpus uses curly quotes/apostrophes); decode uses `errors="replace"`.
  Normalising the corpus to ASCII punctuation would largely remove this.
- **KV-caching** is correct and reduces per-token *compute* to O(1), but at this scale
  (small model, ≤512 ctx, GPU) PyTorch's fused flash kernel makes full-sequence recompute
  about as fast, so it's parity here; the wall-clock win appears with larger models,
  longer contexts, or CPU inference.

---

## Configuration Reference

`configs/shakespeare.yaml`:

```yaml
model:
  vocab_size: 10000       # BPE vocabulary size
  d_model: 512            # embedding dimension
  n_heads: 8              # attention heads (d_model must be divisible by n_heads)
  n_layers: 6             # number of transformer blocks
  d_ff: 1536              # SwiGLU hidden dim
  max_seq_len: 512        # context window length
  dropout: 0.1
  rope_theta: 10000.0     # RoPE base frequency

training:
  batch_size: 64
  learning_rate: 6.0e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  warmup_steps: 200
  max_steps: 8000
  eval_interval: 500
  eval_steps: 100
  checkpoint_dir: checkpoints/
  log_interval: 50
  label_smoothing: 0.0
  early_stopping_patience: 10
  grad_accum_steps: 1     # raise to simulate a larger batch
  use_amp: true           # bf16 autocast on CUDA
  compile: false          # torch.compile (leave off on Windows unless Triton is set up)

data:
  train_file: data/train.txt
  val_file: data/val.txt
  tokenizer_path: tokenizer/bpe.json

device: cuda              # falls back to cpu if CUDA unavailable
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover attention correctness (causal masking, shapes, gradient flow), BPE
encode/decode round-trips and merge ordering, embedding dimensions, and full transformer
forward/generate behaviour + parameter count.

Run the architecture / overfitting diagnostic:

```bash
python check_overfit.py
```

---

## Monitoring

If you have [MiniFlow](https://github.com/Mannava-Daasaradhi/miniflow) installed:

```bash
miniflow runs list                              # all runs
miniflow runs best --metric val_loss --mode min # best checkpoint
miniflow models list                            # registered model snapshots
```

---

## Changelog — Modernisation Pass

| Area | Change | Reason |
|---|---|---|
| Tokenizer | `encode` O(words × merges) → rank-based + per-word cache | Was pathologically slow; now ~instant (identical output) |
| Tokenizer | whitespace-split → **byte-level BPE + regex pre-tokenizer** | Preserves newlines/whitespace; exact round-trip; generated text now has line structure |
| Attention | 5000×5000 mask buffer → `F.scaled_dot_product_attention(is_causal=True)` | Flash kernel; O(T) memory; checkpoints shrank from 888 MB |
| Positional | Sinusoidal additive PE → **RoPE** | Relative positions, no params |
| Norm | LayerNorm → **RMSNorm** | Cheaper, no bias |
| FFN | GELU MLP → **SwiGLU** | Stronger gated FFN |
| Data | 1.1 MB Shakespeare → **24 MB** of 25 classics | Justifies the 10K vocab; fixes the capacity/data mismatch |
| Training | Fixed windows → random-offset sampling; token `.bin` cache; bf16; fused AdamW | Faster, better data utilisation |
| Metrics | Added **next-token accuracy** | Directly measures prediction quality |
| Checkpoints | Lean (weights + config only) | No optimizer state or mask buffers |
| Regularisation | dropout 0.4 → 0.1, label smoothing 0.1 → 0.0 | Big corpus removes the need for heavy regularisation |
| Context | max_seq_len 256 → **512** | Longer-range coherence |
| Generation | added **KV-caching** (`use_cache`, default on) | O(1)-compute per-token decode, output identical to cache-free path (wall-clock win realised at larger scale — see note) |
