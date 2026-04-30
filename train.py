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

# --- Fallback for ExperimentTracker ---
try:
    from miniflow import ExperimentTracker
except ImportError:
    print("Warning: miniflow not found. Using dummy ExperimentTracker.")
    class ExperimentTracker:
        def __init__(self, name): pass
        def log_params(self, params): pass
        def log_metric(self, name, value, step): pass
        def finish(self): pass

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
        self.stride = seq_len // 2

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
        
        # No decay for biases, LayerNorms, or Embeddings
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
    
    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# --- Evaluation & Saving ---
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

def save_checkpoint(model, optimizer, step, val_loss, config, best_val_loss):
    """Save to checkpoints/step_{step}.pt. Keep only best (lowest val_loss)."""
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    
    if val_loss < best_val_loss:
        ckpt_path = os.path.join(config.training.checkpoint_dir, "best.pt")
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config # Save config so generation script knows the architecture
        }, ckpt_path)
        print(f"  --> Saved new best checkpoint to {ckpt_path}")
        return val_loss
    return best_val_loss


# --- Main Loop ---
def train(config_path: str):
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = ConfigNode(config_dict)

    # Setup
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
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
    tokenizer = BPETokenizer.load(config.data.tokenizer_path)

    # Build datasets + dataloaders
    train_ds = TextDataset(config.data.train_file, tokenizer, config.model.max_seq_len)
    val_ds = TextDataset(config.data.val_file, tokenizer, config.model.max_seq_len)

    # pin_memory=True speeds up CPU to GPU data transfer
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, pin_memory=True)
    
    # Creates an infinite iterator over the dataloader
    train_iter = itertools.cycle(train_loader)

    # Build model → move to device
    model = GPT(config.model).to(device)
    print(f"Model parameters: {model.get_num_params():,}")

    # Configure optimizer
    optimizer = configure_optimizer(model, config)
    
    best_val_loss = float('inf')

    # Training loop
    for step in range(config.training.max_steps):
        # Set LR for this step
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward
        model.train()
        x, y = next(train_iter)
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
        if step > 0 and step % config.training.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, config.training.eval_steps)
            val_ppl = math.exp(val_loss)
            
            tracker.log_metric("val_loss", val_loss, step=step)
            tracker.log_metric("val_perplexity", val_ppl, step=step)
            print(f"  VAL | loss={val_loss:.4f} | perplexity={val_ppl:.2f}")

            # Save checkpoint if best so far
            best_val_loss = save_checkpoint(model, optimizer, step, val_loss, config, best_val_loss)

    tracker.finish()
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    train(args.config)