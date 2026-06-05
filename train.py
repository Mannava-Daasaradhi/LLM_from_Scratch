"""
Full training script. Run as:
  python train.py --config configs/shakespeare.yaml
"""

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import argparse
import math
import os
import time

from tokenizer.bpe import BPETokenizer
from model.transformer import GPT
from miniflow import ExperimentTracker, ModelRegistry

PAD_TOKEN_ID = 0  # matches BPE tokenizer convention


# --- Dynamic Config Loader ---
class ConfigNode:
    """Helper class to access dict keys via dot notation (e.g., config.model.d_model)"""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, ConfigNode(v) if isinstance(v, dict) else v)


# --- Tokenised data (cached to .bin so we encode the corpus only once) ---
def load_tokens(filepath: str, tokenizer, tokenizer_path: str = None) -> np.ndarray:
    """
    Encode `filepath` to a flat uint16 token array, cached next to it as `<file>.bin`.
    Re-encodes if the cache is missing or older than the source text OR the tokenizer
    (so changing the tokenizer correctly invalidates stale token streams).
    uint16 is safe because vocab_size (10 000) < 65 536.
    """
    cache_path = filepath + ".bin"
    if os.path.exists(cache_path):
        fresh = os.path.getmtime(cache_path) >= os.path.getmtime(filepath)
        if tokenizer_path and os.path.exists(tokenizer_path):
            fresh = fresh and os.path.getmtime(cache_path) >= os.path.getmtime(tokenizer_path)
        if fresh:
            return np.fromfile(cache_path, dtype=np.uint16)

    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    t0 = time.time()
    ids = tokenizer.encode(text, add_special_tokens=False)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(cache_path)
    print(f"  Tokenised {filepath}: {len(text):,} chars -> {len(arr):,} tokens "
          f"in {time.time() - t0:.1f}s (cached to {cache_path})")
    return arr


def get_batch(data: np.ndarray, batch_size: int, seq_len: int, device) -> tuple:
    """
    Sample `batch_size` random windows of length `seq_len` from the token stream.
    Random offsets (nanoGPT-style) give endless varied batches and far better data
    utilisation than fixed non-overlapping windows.
    """
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# --- Optimizer ---
def configure_optimizer(model, config):
    """
    Weight decay on weight matrices only — not biases, norms, or embeddings.
    """
    decay_params, no_decay_params = [], []
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if pn.endswith("bias") or "norm" in pn or "emb" in pn:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": config.training.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    # fused AdamW is faster on CUDA
    use_fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        param_groups,
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        fused=use_fused,
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
def evaluate(model, val_data, config, device, amp_ctx) -> tuple[float, float]:
    """
    Average val loss (clean cross-entropy) and next-token top-1 accuracy over
    eval_steps random batches. Accuracy directly answers 'how often does the model
    predict the correct next token', which loss/perplexity only describe indirectly.
    """
    model.eval()
    bs, sl = config.training.batch_size, config.model.max_seq_len
    losses, correct, total = [], 0, 0
    for _ in range(config.training.eval_steps):
        x, y = get_batch(val_data, bs, sl, device)
        with amp_ctx:
            logits, loss = model(x, targets=y)
        losses.append(loss.item())
        preds = logits.argmax(dim=-1)
        mask = y != PAD_TOKEN_ID
        correct += (preds[mask] == y[mask]).sum().item()
        total += mask.sum().item()
    model.train()
    return sum(losses) / len(losses), (correct / total if total else 0.0)


# --- Checkpoint + Registry ---
def _clean_state_dict(model) -> dict:
    """Unwrap torch.compile (`_orig_mod.` prefix) so checkpoints load anywhere."""
    raw = getattr(model, "_orig_mod", model)
    return raw.state_dict()


def save_checkpoint(model, step, val_loss, val_acc, config_dict, best_val_loss,
                    tracker_run_id: str) -> float:
    """
    Save a *lean* best checkpoint (model weights + config + metadata only — no
    optimizer state and no attention-mask buffers, so the file is small) and
    register it with MiniFlow. Returns the updated best_val_loss.
    """
    if val_loss >= best_val_loss:
        return best_val_loss

    os.makedirs(config_dict["training"]["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(config_dict["training"]["checkpoint_dir"], "best.pt")
    torch.save({
        "step": step,
        "model_state_dict": _clean_state_dict(model),
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "config": config_dict,
    }, ckpt_path)
    print(f"  --> Saved new best checkpoint to {ckpt_path}")

    registry = ModelRegistry()
    model_id = registry.save(
        name="gpt_classics",
        model_obj=getattr(model, "_orig_mod", model),
        metadata={
            "step": step,
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(math.exp(val_loss), 4),
            "val_accuracy": round(val_acc, 4),
            "run_id": tracker_run_id,
            "checkpoint_path": ckpt_path,
        },
    )
    print(f"  --> Registered model as '{model_id}' in MiniFlow registry")
    return val_loss


# --- Main Loop ---
def train(config_path: str):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = ConfigNode(config_dict)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True   # faster matmuls
        torch.backends.cudnn.allow_tf32 = True

    # Mixed precision context (bf16 on CUDA needs no GradScaler)
    use_amp = getattr(config.training, "use_amp", True) and device.type == "cuda"
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp else torch.autocast(device_type="cpu", enabled=False))
    print(f"Mixed precision (bf16 autocast): {use_amp}")

    # MiniFlow run
    tracker = ExperimentTracker("llm_classics")
    tracker.log_params({
        "d_model": config.model.d_model, "n_layers": config.model.n_layers,
        "n_heads": config.model.n_heads, "vocab_size": config.model.vocab_size,
        "max_seq_len": config.model.max_seq_len, "dropout": config.model.dropout,
        "batch_size": config.training.batch_size, "max_steps": config.training.max_steps,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "warmup_steps": config.training.warmup_steps,
        "grad_clip": config.training.grad_clip,
    })
    print(f"MiniFlow run started: {tracker.run_id}")

    # Data
    tokenizer = BPETokenizer.load(config.data.tokenizer_path)
    train_data = load_tokens(config.data.train_file, tokenizer, config.data.tokenizer_path)
    val_data = load_tokens(config.data.val_file, tokenizer, config.data.tokenizer_path)
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    # Model
    model = GPT(config.model).to(device)
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,}")
    tracker.log_params({"num_params": num_params})

    optimizer = configure_optimizer(model, config)

    if getattr(config.training, "compile", False) and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}); continuing uncompiled")

    grad_accum = max(1, getattr(config.training, "grad_accum_steps", 1))
    label_smoothing = getattr(config.training, "label_smoothing", 0.0)
    patience = getattr(config.training, "early_stopping_patience", 999999)
    print(f"grad_accum_steps={grad_accum} | label_smoothing={label_smoothing} | "
          f"early_stopping_patience={patience}")

    best_val_loss = float("inf")
    patience_counter = 0
    bs, sl = config.training.batch_size, config.model.max_seq_len
    t_log = time.time()

    for step in range(config.training.max_steps):
        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        # Gradient accumulation: split the effective batch into grad_accum micro-batches
        for _ in range(grad_accum):
            x, y = get_batch(train_data, bs, sl, device)
            with amp_ctx:
                logits, _ = model(x, targets=None)
                loss = F.cross_entropy(
                    logits.view(-1, config.model.vocab_size), y.view(-1),
                    ignore_index=PAD_TOKEN_ID, label_smoothing=label_smoothing,
                )
            (loss / grad_accum).backward()
            loss_accum += loss.item() / grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()

        if step % config.training.log_interval == 0:
            dt = time.time() - t_log
            t_log = time.time()
            tracker.log_metric("train_loss", loss_accum, step=step)
            tracker.log_metric("lr", lr, step=step)
            print(f"Step {step:5d} | loss={loss_accum:.4f} | lr={lr:.2e} | "
                  f"{dt / max(1, config.training.log_interval) * 1000:.0f} ms/step")

        if step > 0 and step % config.training.eval_interval == 0:
            val_loss, val_acc = evaluate(model, val_data, config, device, amp_ctx)
            val_ppl = math.exp(val_loss)
            tracker.log_metric("val_loss", val_loss, step=step)
            tracker.log_metric("val_perplexity", val_ppl, step=step)
            tracker.log_metric("val_accuracy", val_acc, step=step)
            print(f"  VAL | loss={val_loss:.4f} | perplexity={val_ppl:.2f} | "
                  f"next-token acc={val_acc * 100:.2f}%")

            prev_best = best_val_loss
            best_val_loss = save_checkpoint(
                model, step, val_loss, val_acc, config_dict, best_val_loss, tracker.run_id
            )
            if val_loss < prev_best:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter}/{patience} eval checks")
                if patience_counter >= patience:
                    print(f"  Early stopping at step {step}. Best val loss: {best_val_loss:.4f}")
                    tracker.log_metric("stopped_early_at_step", step, step=step)
                    tracker.finish()
                    return

    tracker.finish()
    print(f"Training complete! Run ID: {tracker.run_id}")
    print(f"View results: miniflow runs best --metric val_loss --mode min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
