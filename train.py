"""
Full training script. Run as:
  python train.py --config configs/shakespeare.yaml
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
import argparse
import math
import os
import itertools
from pathlib import Path

from tokenizer.bpe import BPETokenizer
from model.transformer import GPT
from miniflow import ExperimentTracker, ModelRegistry


# --- Dynamic Config Loader ---
class ConfigNode:
    """Helper class to access dict keys via dot notation (e.g., config.model.d_model)"""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, ConfigNode(v) if isinstance(v, dict) else v)


# --- Dataset ---
class TextDataset(Dataset):
    """
    Loads tokenized text and returns (input_ids, targets) pairs.
    input_ids:  tokens[i : i + seq_len]
    targets:    tokens[i+1 : i + seq_len + 1]  (next-token prediction)
    Stride: seq_len // 2 (50% overlap between consecutive windows — more training examples)
    """
    def __init__(self, filepath: str, tokenizer, seq_len: int):
        with open(filepath, encoding='utf-8') as f:
            text = f.read()
        self.tokens = torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
        self.seq_len = seq_len
        # CHANGED: stride from seq_len//2 → seq_len (non-overlapping windows).
        # 50% overlap meant ~2x duplicate token context per training step,
        # causing the model to memorise specific windows and overfit.
        self.stride = seq_len

    def __len__(self):
        return (len(self.tokens) - self.seq_len - 1) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y


# --- Optimizer ---
def configure_optimizer(model, config):
    """
    Apply weight decay only to weight matrices, NOT to:
    - Bias terms
    - LayerNorm weights and biases
    - Embedding weights
    """
    decay_params = []
    no_decay_params = []

    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if pn.endswith('bias') or 'norm' in pn or 'emb' in pn or 'pe' in pn:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": config.training.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2)
    )


# --- Learning Rate Scheduler ---
def get_lr(step: int, config) -> float:
    """Linear warmup for warmup_steps, then cosine decay to 10% of max LR."""
    max_lr = config.training.learning_rate
    min_lr = max_lr * 0.1
    warmup_steps = config.training.warmup_steps
    max_steps = config.training.max_steps

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# --- Evaluation ---
@torch.no_grad()
def evaluate(model, val_loader, device, n_steps) -> float:
    """Average val loss over n_steps batches."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= n_steps:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


# --- Checkpoint + Registry ---
def save_checkpoint(model, optimizer, step, val_loss, config, best_val_loss, tracker_run_id: str) -> float:
    """
    Saves best checkpoint to disk via torch and registers it with
    MiniFlow ModelRegistry so it appears under `miniflow models list`.
    Returns updated best_val_loss.
    """
    if val_loss >= best_val_loss:
        return best_val_loss

    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(config.training.checkpoint_dir, "best.pt")

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config,
    }, ckpt_path)
    print(f"  --> Saved new best checkpoint to {ckpt_path}")

    # Register the model with MiniFlow so it's queryable via CLI
    registry = ModelRegistry()
    model_id = registry.save(
        name="gpt_shakespeare",
        model_obj=model,
        metadata={
            "step": step,
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(math.exp(val_loss), 4),
            "run_id": tracker_run_id,
            "checkpoint_path": ckpt_path,
        }
    )
    print(f"  --> Registered model as '{model_id}' in MiniFlow registry")

    return val_loss


# --- Main Loop ---
def train(config_path: str):
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = ConfigNode(config_dict)

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Start MiniFlow run — this creates the DB row immediately
    tracker = ExperimentTracker("llm_shakespeare")
    tracker.log_params({
        "d_model": config.model.d_model,
        "n_layers": config.model.n_layers,
        "n_heads": config.model.n_heads,
        "vocab_size": config.model.vocab_size,
        "batch_size": config.training.batch_size,
        "max_steps": config.training.max_steps,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "warmup_steps": config.training.warmup_steps,
        "grad_clip": config.training.grad_clip,
    })
    print(f"MiniFlow run started: {tracker.run_id}")

    # Load tokenizer
    tokenizer = BPETokenizer.load(config.data.tokenizer_path)

    # Build datasets + dataloaders
    train_ds = TextDataset(config.data.train_file, tokenizer, config.model.max_seq_len)
    val_ds = TextDataset(config.data.val_file, tokenizer, config.model.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, pin_memory=True)
    train_iter = itertools.cycle(train_loader)

    # Build model
    model = GPT(config.model).to(device)
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,}")
    tracker.log_params({"num_params": num_params})

    optimizer = configure_optimizer(model, config)
    best_val_loss = float('inf')

    # ADDED: Early stopping state
    # patience: how many consecutive val checks with no improvement before stopping
    # Read from config if present, otherwise default to 3
    PATIENCE = getattr(config.training, 'early_stopping_patience', 3)
    patience_counter = 0
    print(f"Early stopping patience: {PATIENCE} eval intervals")

    # ADDED: Label smoothing value — read from config if present, default 0.1
    # Prevents the model from becoming overconfident on training tokens.
    # softmax target distribution becomes (1 - ls) for the true token
    # and ls / (vocab_size - 1) for all other tokens instead of a one-hot.
    LABEL_SMOOTHING = getattr(config.training, 'label_smoothing', 0.1)
    PAD_TOKEN_ID = 0  # matches BPE tokenizer convention
    print(f"Label smoothing: {LABEL_SMOOTHING}")

    # Training loop
    for step in range(config.training.max_steps):
        # Update LR
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward
        model.train()
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # ADDED: Get logits only (targets=None), then compute loss here so we
        # can pass label_smoothing.  The model's internal loss path stays intact
        # and is still used during evaluate() — no code removed from model.
        logits, _ = model(x, targets=None)
        loss = F.cross_entropy(
            logits.view(-1, config.model.vocab_size),
            y.view(-1),
            ignore_index=PAD_TOKEN_ID,
            label_smoothing=LABEL_SMOOTHING,   # ADDED
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()

        # Log training metrics
        if step % config.training.log_interval == 0:
            tracker.log_metric("train_loss", loss.item(), step=step)
            tracker.log_metric("lr", lr, step=step)
            print(f"Step {step:5d} | loss={loss.item():.4f} | lr={lr:.2e}")

        # Evaluate + checkpoint
        if step > 0 and step % config.training.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, config.training.eval_steps)
            val_ppl = math.exp(val_loss)

            tracker.log_metric("val_loss", val_loss, step=step)
            tracker.log_metric("val_perplexity", val_ppl, step=step)
            print(f"  VAL | loss={val_loss:.4f} | perplexity={val_ppl:.2f}")

            # FIXED: capture the current best BEFORE save_checkpoint overwrites it,
            # so the patience check below compares against the pre-update best.
            prev_best = best_val_loss
            best_val_loss = save_checkpoint(
                model, optimizer, step, val_loss,
                config, best_val_loss,
                tracker_run_id=tracker.run_id
            )

            # FIXED: was comparing val_loss < best_val_loss AFTER save_checkpoint
            # already updated best_val_loss, so patience_counter always reset to 0
            # even when there was no real improvement — early stopping never fired.
            if val_loss < prev_best:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter}/{PATIENCE} eval checks")
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping triggered at step {step}. Best val loss: {best_val_loss:.4f}")
                    tracker.log_metric("stopped_early_at_step", step, step=step)
                    tracker.finish()
                    print(f"Training stopped early! Run ID: {tracker.run_id}")
                    print(f"View results: miniflow runs best --metric val_loss --mode min")
                    return  # exit cleanly instead of continuing to overfit

    tracker.finish()
    print(f"Training complete! Run ID: {tracker.run_id}")
    print(f"View results: miniflow runs best --metric val_loss --mode min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)