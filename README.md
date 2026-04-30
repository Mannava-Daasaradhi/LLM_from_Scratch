# LLM from Scratch 🧠

A GPT-style language model built entirely from scratch in PyTorch, trained on the Shakespeare dataset. This project implements every component — BPE tokenizer, multi-head causal self-attention, transformer blocks, and a full training loop with regularisation and early stopping.

---

## Project Structure

```
LLM_from_Scratch/
├── config.py                  # Typed config dataclasses + YAML loader
├── configs/
│   └── shakespeare.yaml       # All hyperparameters in one place
├── data/
│   ├── download.py            # Download the Shakespeare corpus
│   └── input.txt              # Raw text (train/val split at runtime)
├── model/
│   ├── attention.py           # Multi-head causal self-attention
│   ├── block.py               # Transformer block (pre-norm + residuals)
│   ├── embedding.py           # Token + sinusoidal positional embeddings
│   ├── feedforward.py         # Position-wise FFN (GELU activation)
│   └── transformer.py         # Full GPT model + generation
├── tokenizer/
│   ├── bpe.py                 # Byte-Pair Encoding tokenizer
│   ├── train_tokenizer.py     # Train a BPE vocab from text
│   └── base.py                # Abstract base class
├── plots/
│   └── plot_curves.py         # Plot train/val loss curves from logs
├── tests/                     # Unit tests for each component
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

### 2. Prepare data

```bash
python data/download.py          # downloads Shakespeare corpus to data/input.txt
# Then split into train/val (e.g. 90/10):
python -c "
text = open('data/input.txt').read()
n = int(len(text) * 0.9)
open('data/train.txt','w').write(text[:n])
open('data/val.txt','w').write(text[n:])
print('Done.')
"
```

### 3. Train the BPE tokenizer

```bash
python tokenizer/train_tokenizer.py \
    --input data/train.txt \
    --vocab-size 10000 \
    --output tokenizer/shakespeare_bpe.json
```

### 4. Train the model

```bash
python train.py --config configs/shakespeare.yaml
```

Training logs are printed every `log_interval` steps. Validation is run every `eval_interval` steps. The best checkpoint is saved to `checkpoints/best.pt`.

### 5. Generate text

```bash
python generate.py \
    --checkpoint checkpoints/best.pt \
    --prompt "To be or not to be" \
    --max-tokens 200 \
    --temperature 0.8 \
    --top-k 40
```

---

## Model Architecture

This is a decoder-only transformer (GPT-style) with the following design choices:

| Component | Choice | Why |
|---|---|---|
| Positional encoding | Sinusoidal (fixed) | No extra parameters; generalises to unseen lengths |
| Attention | Multi-head causal self-attention | Prevents attending to future tokens |
| Normalisation | Pre-LayerNorm | Stable gradient flow through deep networks |
| Activation | GELU | Smoother than ReLU; empirically better for LMs |
| Weight tying | Token emb ↔ LM head | Reduces ~5M params; improves generalisation |
| Init | N(0, 0.02) + residual scaling | GPT-2 convention; controls residual stream growth |

Default hyperparameters (`configs/shakespeare.yaml`):

```
d_model:     512    n_heads:  8    n_layers: 6
d_ff:       2048    dropout: 0.4   max_seq_len: 256
vocab_size: 10000
~38M parameters
```

---

## Training Details

### Optimiser
AdamW with selective weight decay — weight matrices decay, biases and LayerNorm parameters do not.

### Learning Rate Schedule
Linear warmup for `warmup_steps` steps, then cosine decay to 10% of peak LR.

### Anti-Overfitting Measures

Four complementary techniques are applied. All are configurable via `configs/shakespeare.yaml`:

#### 1. Dropout (`model.dropout = 0.4`)
Applied inside attention (on attention weights) and after each FFN activation. Randomly zeroes activations during training, preventing co-adaptation of features.

#### 2. Label Smoothing (`training.label_smoothing = 0.1`)
Instead of training against a one-hot target, the true token receives probability `1 - ls = 0.9` and the remaining `0.1` is spread uniformly across all other tokens. This prevents the model from becoming overconfident on training tokens. Applied **only to training loss** — val loss uses clean cross-entropy so it remains interpretable.

#### 3. Weight Decay (`training.weight_decay = 0.1`)
L2 regularisation on weight matrices. Keeps weights small and discourages memorisation. Set to `0.1` (down from the original `0.3` which was too aggressive and harmed generalisation).

#### 4. Early Stopping (`training.early_stopping_patience = 4`)
Training stops automatically if validation loss does not improve for `patience` consecutive evaluation intervals. This prevents the model from continuing to fit training noise after it has already started to diverge on held-out data.

#### 5. Non-Overlapping Data Windows
`TextDataset` uses `stride = seq_len` (non-overlapping windows). The original 50% overlap (`stride = seq_len // 2`) created near-duplicate training samples which amplified memorisation.

---

## Configuration Reference

`configs/shakespeare.yaml`:

```yaml
model:
  vocab_size: 10000       # BPE vocabulary size
  d_model: 512            # embedding dimension
  n_heads: 8              # attention heads (d_model must be divisible by n_heads)
  n_layers: 6             # number of transformer blocks
  d_ff: 2048              # FFN hidden dim (typically 4 × d_model)
  max_seq_len: 256        # context window length
  dropout: 0.4            # dropout rate (applied in attention + FFN)

training:
  batch_size: 64
  learning_rate: 3.0e-4
  weight_decay: 0.1       # L2 regularisation on weight matrices only
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0          # gradient norm clipping threshold
  warmup_steps: 100
  max_steps: 5000
  eval_interval: 500      # run validation every N steps
  eval_steps: 100         # average val loss over this many batches
  checkpoint_dir: checkpoints/
  log_interval: 50
  label_smoothing: 0.1    # training-time label smoothing strength
  early_stopping_patience: 4  # stop after N evals with no improvement

data:
  train_file: data/train.txt
  val_file: data/val.txt
  tokenizer_path: tokenizer/shakespeare_bpe.json

device: cuda              # falls back to cpu if CUDA unavailable
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover attention correctness (causal masking, shape), BPE encode/decode round-trips, embedding dimensions, and full transformer forward pass shapes.

---

## Monitoring

If you have [MiniFlow](https://github.com/Mannava-Daasaradhi/miniflow) installed:

```bash
miniflow runs list                              # all runs
miniflow runs best --metric val_loss --mode min # best checkpoint
miniflow models list                            # registered model snapshots
```

---

## Changes from Original (Anti-Overfitting Fixes)

| File | Change | Reason |
|---|---|---|
| `configs/shakespeare.yaml` | `dropout` 0.3 → **0.4** | Stronger regularisation |
| `configs/shakespeare.yaml` | `weight_decay` 0.3 → **0.1** | 0.3 was too aggressive; hurt generalisation |
| `configs/shakespeare.yaml` | Added `label_smoothing: 0.1` | Config-driven smoothing strength |
| `configs/shakespeare.yaml` | Added `early_stopping_patience: 4` | Config-driven patience |
| `train.py` | `stride` = `seq_len//2` → **`seq_len`** | Removes near-duplicate training windows |
| `train.py` | Fixed early stopping `prev_best` bug | Counter always reset before; stopping never triggered |
| `model/transformer.py` | Removed `label_smoothing=0.1` from `forward()` | Val loss must be clean cross-entropy; smoothing only belongs in the training loss path |