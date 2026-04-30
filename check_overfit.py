"""
Overfitting diagnostic for LLM_from_Scratch.
Run as:  python check_overfit.py [--checkpoint checkpoints/best.pt]

What it checks
──────────────
1. SINGLE-BATCH MEMORISATION
   Trains the model on ONE fixed batch for 200 steps with dropout=0.
   Loss must reach near-zero (< 0.5). If it can't, the model has a
   gradient/architecture bug — not an overfitting problem.

2. TRAIN vs VAL GAP (generalisation check)
   Trains on a small random train split and evaluates on a held-out
   val split that the model has never seen.  Reports:
     • train loss at the end of training
     • val loss at the end of training
     • gap = val_loss - train_loss
   A gap > 1.0 is a strong signal of overfitting.
   A gap ≤ 0.5 on this toy run indicates healthy generalisation.

3. LOSS TREND (is val loss going up while train loss goes down?)
   Samples loss every 25 steps. Prints a mini ASCII chart and flags
   if the final 25% of val-loss readings trend upward while train
   loss is still falling.

4. REGULARISATION AUDIT
   Reads configs/shakespeare.yaml and checks that anti-overfitting
   knobs are set to safe values:
     • dropout          ≥ 0.3
     • weight_decay     ≤ 0.2  (above 0.2 hurts generalisation)
     • label_smoothing  ≥ 0.05
     • early_stopping_patience is present

5. ARCHITECTURE SANITY
   Verifies weight tying and that the model can back-prop without
   NaN gradients.

Usage
─────
  python check_overfit.py                     # runs all checks on tiny synthetic data
  python check_overfit.py --steps 300         # more training steps for checks 2 & 3
  python check_overfit.py --seed 0            # fix random seed
"""

import argparse
import math
import sys
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

# ── make sure the repo root is on PYTHONPATH when run from any cwd ──
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model.transformer import GPT


# ─────────────────────────────────────────────
#  Tiny model config used for all in-memory checks
# ─────────────────────────────────────────────
@dataclass
class TinyConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 64
    dropout: float = 0.0   # dropout=0 for memorisation test; re-enabled for gap test


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
INFO = "\033[94mℹ INFO\033[0m"

def banner(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def make_batch(vocab_size, seq_len, batch_size, device):
    x = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    return x, y

def train_steps(model, x, y, n_steps, lr=1e-3, label_smoothing=0.0):
    """Return list of loss values, one per step."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(n_steps):
        opt.zero_grad()
        logits, _ = model(x, targets=None)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=0,
            label_smoothing=label_smoothing,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses

@torch.no_grad()
def eval_loss(model, x, y):
    model.eval()
    logits, _ = model(x, targets=None)
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
        ignore_index=0,
    ).item()

def ascii_chart(values, width=50, label=""):
    """Print a tiny ASCII sparkline."""
    lo, hi = min(values), max(values)
    rng = hi - lo if hi != lo else 1.0
    bars = "▁▂▃▄▅▆▇█"
    line = "".join(bars[min(7, int((v - lo) / rng * 7))] for v in values)
    print(f"  {label:12s} [{line}]  min={lo:.3f}  max={hi:.3f}")


# ─────────────────────────────────────────────
#  Check 1 — Single-batch memorisation
# ─────────────────────────────────────────────
def check_memorisation(device, steps=200):
    banner("CHECK 1 — Single-batch memorisation")
    print("  Trains on ONE fixed batch for %d steps (dropout=0)." % steps)
    print("  Loss should reach < 0.5 if the model can learn at all.\n")

    cfg = TinyConfig(dropout=0.0)
    model = GPT(cfg).to(device)
    x, y = make_batch(cfg.vocab_size, cfg.seq_len, batch_size=4, device=device)

    losses = train_steps(model, x, y, n_steps=steps, lr=5e-3)
    final = losses[-1]
    initial = losses[0]

    print(f"  Initial loss : {initial:.4f}")
    print(f"  Final loss   : {final:.4f}  (target < 0.5)")
    ascii_chart(losses[::max(1, steps//50)], label="train loss")

    if final < 0.5:
        print(f"\n  {PASS}  Model can memorise a single batch (loss={final:.4f})")
    else:
        print(f"\n  {FAIL}  Model could NOT memorise a single batch (loss={final:.4f})")
        print("         This suggests a bug in the architecture or optimizer, not overfitting.")
    return final < 0.5


# ─────────────────────────────────────────────
#  Check 2 — Train / Val gap
# ─────────────────────────────────────────────
def check_train_val_gap(device, steps=200, label_smoothing=0.1, dropout=0.4):
    banner("CHECK 2 — Train vs Val generalisation gap")
    print(f"  dropout={dropout}, label_smoothing={label_smoothing}, steps={steps}")
    print("  Gap = val_loss - train_loss.  Gap > 1.0 = overfitting signal.\n")

    cfg = TinyConfig(dropout=dropout)
    model = GPT(cfg).to(device)

    x_train, y_train = make_batch(cfg.vocab_size, cfg.seq_len, batch_size=16, device=device)
    x_val,   y_val   = make_batch(cfg.vocab_size, cfg.seq_len, batch_size=16, device=device)

    train_losses, val_losses, checkpoints = [], [], []

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    record_every = max(1, steps // 20)

    model.train()
    for step in range(steps):
        opt.zero_grad()
        logits, _ = model(x_train, targets=None)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y_train.view(-1),
            ignore_index=0,
            label_smoothing=label_smoothing,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % record_every == 0:
            tl = eval_loss(model, x_train, y_train)
            vl = eval_loss(model, x_val,   y_val)
            train_losses.append(tl)
            val_losses.append(vl)
            checkpoints.append(step)
            model.train()

    final_train = train_losses[-1]
    final_val   = val_losses[-1]
    gap = final_val - final_train

    ascii_chart(train_losses, label="train loss")
    ascii_chart(val_losses,   label="val loss  ")
    print()
    print(f"  Final train loss : {final_train:.4f}")
    print(f"  Final val loss   : {final_val:.4f}")
    print(f"  Gap (val - train): {gap:+.4f}")

    if gap <= 0.5:
        print(f"\n  {PASS}  Gap {gap:+.4f} ≤ 0.5 — good generalisation")
    elif gap <= 1.0:
        print(f"\n  {WARN}  Gap {gap:+.4f} is moderate — monitor during full training")
    else:
        print(f"\n  {FAIL}  Gap {gap:+.4f} > 1.0 — model is overfitting")

    return gap, train_losses, val_losses


# ─────────────────────────────────────────────
#  Check 3 — Val loss trend
# ─────────────────────────────────────────────
def check_val_trend(val_losses):
    banner("CHECK 3 — Val loss trend")
    n = len(val_losses)
    if n < 4:
        print(f"  {INFO}  Not enough checkpoints to trend-check (need ≥ 4, got {n})")
        return True

    # Compare first quarter average vs last quarter average
    q = max(1, n // 4)
    early_avg = sum(val_losses[:q]) / q
    late_avg  = sum(val_losses[-q:]) / q

    print(f"  Early val avg (first {q} pts) : {early_avg:.4f}")
    print(f"  Late  val avg (last  {q} pts) : {late_avg:.4f}")

    if late_avg < early_avg:
        print(f"\n  {PASS}  Val loss is trending DOWN — training is generalising")
        return True
    else:
        delta = late_avg - early_avg
        print(f"\n  {FAIL}  Val loss trending UP by {delta:.4f} — overfitting in progress")
        return False


# ─────────────────────────────────────────────
#  Check 4 — Regularisation audit
# ─────────────────────────────────────────────
def check_regularisation_config():
    banner("CHECK 4 — Regularisation config audit")
    cfg_path = ROOT / "configs" / "shakespeare.yaml"
    if not cfg_path.exists():
        print(f"  {WARN}  {cfg_path} not found — skipping")
        return

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg    = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    checks = []

    # dropout
    dropout = model_cfg.get("dropout", None)
    if dropout is None:
        checks.append((False, f"dropout not set in model config"))
    elif dropout >= 0.3:
        checks.append((True,  f"dropout = {dropout} (≥ 0.3)"))
    else:
        checks.append((False, f"dropout = {dropout} — too low (raise to ≥ 0.3)"))

    # weight_decay
    wd = training_cfg.get("weight_decay", None)
    if wd is None:
        checks.append((False, "weight_decay not set"))
    elif wd <= 0.2:
        checks.append((True,  f"weight_decay = {wd} (≤ 0.2)"))
    else:
        checks.append((False, f"weight_decay = {wd} — too high (lower to ≤ 0.2, original bug was 0.3)"))

    # label_smoothing
    ls = training_cfg.get("label_smoothing", None)
    if ls is None:
        checks.append((False, "label_smoothing missing — add 'label_smoothing: 0.1' to training config"))
    elif ls >= 0.05:
        checks.append((True,  f"label_smoothing = {ls} (≥ 0.05)"))
    else:
        checks.append((False, f"label_smoothing = {ls} — too low (raise to ≥ 0.05)"))

    # early_stopping_patience
    esp = training_cfg.get("early_stopping_patience", None)
    if esp is None:
        checks.append((False, "early_stopping_patience missing — add it to training config"))
    else:
        checks.append((True,  f"early_stopping_patience = {esp}"))

    all_pass = True
    for ok, msg in checks:
        tag = PASS if ok else FAIL
        print(f"  {tag}  {msg}")
        if not ok:
            all_pass = False

    return all_pass


# ─────────────────────────────────────────────
#  Check 5 — Architecture sanity
# ─────────────────────────────────────────────
def check_architecture_sanity(device):
    banner("CHECK 5 — Architecture sanity")
    cfg = TinyConfig(dropout=0.0)
    model = GPT(cfg).to(device)

    results = []

    # Weight tying
    tied = model.token_emb.embedding.weight.data_ptr() == model.head.weight.data_ptr()
    results.append((tied, "Token embedding and LM head weights are tied"))

    # No NaN gradients after one backward pass
    x, y = make_batch(cfg.vocab_size, cfg.seq_len, batch_size=2, device=device)
    logits, _ = model(x, targets=None)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1), ignore_index=0)
    loss.backward()
    nan_grads = any(
        p.grad is not None and torch.isnan(p.grad).any()
        for p in model.parameters()
    )
    results.append((not nan_grads, "No NaN gradients after backward pass"))

    # Output shape
    model.zero_grad()
    with torch.no_grad():
        out, _ = model(x)
    correct_shape = out.shape == (2, cfg.seq_len, cfg.vocab_size)
    results.append((correct_shape, f"Output shape correct: {tuple(out.shape)}"))

    # Param count > 0
    n = model.get_num_params()
    results.append((n > 0, f"Trainable parameters: {n:,}"))

    all_pass = True
    for ok, msg in results:
        tag = PASS if ok else FAIL
        print(f"  {tag}  {msg}")
        if not ok:
            all_pass = False

    return all_pass


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Overfitting diagnostic for LLM_from_Scratch")
    parser.add_argument("--steps", type=int, default=200,
                        help="Training steps for checks 2 & 3 (default: 200)")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda/cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nRunning on device: {device}  |  seed: {args.seed}  |  steps: {args.steps}")

    results = {}

    # Check 1
    results["memorisation"] = check_memorisation(device, steps=200)

    # Check 2 + 3 share the same training run
    gap, train_losses, val_losses = check_train_val_gap(device, steps=args.steps)
    results["gap_ok"] = gap <= 1.0

    # Check 3
    results["trend_ok"] = check_val_trend(val_losses)

    # Check 4
    results["config_ok"] = check_regularisation_config()

    # Check 5
    results["arch_ok"] = check_architecture_sanity(device)

    # ── Summary ──
    banner("SUMMARY")
    labels = {
        "memorisation": "Check 1 — Single-batch memorisation",
        "gap_ok":       "Check 2 — Train/Val gap ≤ 1.0",
        "trend_ok":     "Check 3 — Val loss trending down",
        "config_ok":    "Check 4 — Regularisation config",
        "arch_ok":      "Check 5 — Architecture sanity",
    }
    all_pass = True
    for key, label in labels.items():
        ok = results.get(key, False)
        tag = PASS if ok else FAIL
        print(f"  {tag}  {label}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  \033[92mAll checks passed — no overfitting detected.\033[0m")
        sys.exit(0)
    else:
        print("  \033[91mSome checks failed — review output above.\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()