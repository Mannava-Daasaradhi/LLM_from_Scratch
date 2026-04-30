"""
plots/plot_curves.py

Pulls train_loss, val_loss, val_perplexity, and lr from MiniFlow
ExperimentTracker and saves a 2-panel training curve to
plots/training_curve.png.

Usage:
    # Latest run (default)
    python plots/plot_curves.py

    # Specific run
    python plots/plot_curves.py --run-id <run_id>

    # Best run by val_loss
    python plots/plot_curves.py --best
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── colour palette ──────────────────────────────────────────────────────────
TRAIN_COLOR   = "#4C72B0"
VAL_COLOR     = "#DD8452"
PPL_COLOR     = "#55A868"
LR_COLOR      = "#C44E52"
GRID_COLOR    = "#E8E8E8"
BG_COLOR      = "#FAFAFA"
SPINE_COLOR   = "#CCCCCC"

def load_run(tracker, run_id: str | None, best: bool) -> tuple[str, dict]:
    if best:
        run = tracker.get_best_run(metric="val_loss", mode="min")
    elif run_id:
        all_runs = tracker.get_runs()
        run = next((r for r in all_runs if r["run_id"].startswith(run_id)), None)
    else:
        run = tracker.get_runs()[0]  # latest first

    if run is None:
        sys.exit(
            "No MiniFlow runs found. Train the model first:\n"
            "  python train.py --config configs/shakespeare.yaml"
        )

    return run["run_id"], run["metrics"]





def _extract(metrics: dict, key: str) -> tuple[list, list]:
    rows = metrics.get(key, [])
    if not rows:
        return [], []
    
    if isinstance(rows[0], (int, float)):
        # infer step interval from key
        if key in ("val_loss", "val_perplexity", "stopped_early_at_step"):
            interval = 500   # eval_interval from config
        elif key == "lr":
            interval = 50    # log_interval from config
        else:
            interval = 50    # train_loss log_interval
        
        steps = [i * interval for i in range(len(rows))]
        return steps, rows
    
    # (step, value) pairs fallback
    rows_sorted = sorted(rows, key=lambda r: r[0])
    return [r[0] for r in rows_sorted], [r[1] for r in rows_sorted]


def plot_curves(run_id: str, metrics: dict, out_path: str) -> None:
    train_steps, train_loss = _extract(metrics, "train_loss")
    val_steps,   val_loss   = _extract(metrics, "val_loss")
    _,           val_ppl    = _extract(metrics, "val_perplexity")
    lr_steps,    lrs        = _extract(metrics, "lr")

    # ── figure layout ────────────────────────────────────────────────────────
    has_lr  = bool(lr_steps)
    n_cols  = 2
    n_rows  = 2 if has_lr else 1
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(13, 9 if has_lr else 5),
        facecolor=BG_COLOR,
    )
    # normalise axes to a flat list so indexing is always [row][col]
    if n_rows == 1:
        axes = [axes]          # shape: [[ax0, ax1]]

    fig.suptitle(
        f"Training Curves  ·  run {run_id}",
        fontsize=14, fontweight="bold", color="#333333", y=0.98,
    )

    # ── helper ───────────────────────────────────────────────────────────────
    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor(BG_COLOR)
        ax.set_title(title, fontsize=11, color="#333333", pad=8)
        ax.set_xlabel(xlabel, fontsize=9,  color="#555555")
        ax.set_ylabel(ylabel, fontsize=9,  color="#555555")
        ax.tick_params(colors="#555555", labelsize=8)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle="--")
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)
            spine.set_linewidth(0.8)

    # ── Panel 1 — Loss ───────────────────────────────────────────────────────
    ax_loss = axes[0][0]
    if train_steps:
        ax_loss.plot(train_steps, train_loss,
                     color=TRAIN_COLOR, linewidth=1.2, alpha=0.85,
                     label="train loss")
    if val_steps:
        ax_loss.plot(val_steps, val_loss,
                     color=VAL_COLOR, linewidth=2.0,
                     marker="o", markersize=4, label="val loss")

        # annotate best val loss
        best_val  = min(val_loss)
        best_step = val_steps[val_loss.index(best_val)]
        ax_loss.annotate(
            f"best {best_val:.4f}",
            xy=(best_step, best_val),
            xytext=(best_step, best_val + (max(val_loss) - min(val_loss)) * 0.12),
            fontsize=7.5, color=VAL_COLOR,
            arrowprops=dict(arrowstyle="->", color=VAL_COLOR, lw=0.8),
        )

    _style(ax_loss, "Loss", "step", "cross-entropy loss")
    ax_loss.legend(fontsize=8, framealpha=0.6)

    # ── Panel 2 — Perplexity ─────────────────────────────────────────────────
    ax_ppl = axes[0][1]
    if val_steps and val_ppl:
        ax_ppl.plot(val_steps, val_ppl,
                    color=PPL_COLOR, linewidth=2.0,
                    marker="s", markersize=4, label="val perplexity")

        # horizontal reference lines
        for ref, label in [(50, "target <50"), (10, "good <10")]:
            if min(val_ppl) < ref * 2:          # only draw if in range
                ax_ppl.axhline(ref, color="#AAAAAA", linewidth=0.9,
                               linestyle=":", zorder=0)
                ax_ppl.text(val_steps[-1], ref + 0.5, label,
                            fontsize=7, color="#888888", ha="right")

        # annotate best
        best_ppl  = min(val_ppl)
        best_step = val_steps[val_ppl.index(best_ppl)]
        ax_ppl.annotate(
            f"best {best_ppl:.2f}",
            xy=(best_step, best_ppl),
            xytext=(best_step, best_ppl + (max(val_ppl) - min(val_ppl)) * 0.12),
            fontsize=7.5, color=PPL_COLOR,
            arrowprops=dict(arrowstyle="->", color=PPL_COLOR, lw=0.8),
        )

    _style(ax_ppl, "Validation Perplexity", "step", "perplexity")
    ax_ppl.legend(fontsize=8, framealpha=0.6)

    # ── Panel 3 — LR schedule (optional) ─────────────────────────────────────
    if has_lr:
        ax_lr = axes[1][0]
        ax_lr.plot(lr_steps, lrs,
                   color=LR_COLOR, linewidth=1.2, alpha=0.9,
                   label="learning rate")
        ax_lr.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))
        _style(ax_lr, "Learning Rate Schedule", "step", "lr")
        ax_lr.legend(fontsize=8, framealpha=0.6)

        # ── Panel 4 — Train vs Val loss zoomed on val checkpoints ─────────────
        ax_zoom = axes[1][1]
        if val_steps:
            # filter train_loss to only steps <= last val step for fair compare
            last = val_steps[-1]
            t_s = [s for s in train_steps if s <= last]
            t_l = [train_loss[i] for i, s in enumerate(train_steps) if s <= last]
            if t_s:
                ax_zoom.plot(t_s, t_l,
                             color=TRAIN_COLOR, linewidth=1.0, alpha=0.6,
                             label="train loss")
            ax_zoom.plot(val_steps, val_loss,
                         color=VAL_COLOR, linewidth=2.0,
                         marker="o", markersize=4, label="val loss")

            # shade overfitting region (val > train)
            if t_s and len(t_s) == len(val_steps):
                overfit = [v - t for v, t in zip(val_loss, t_l)]
                for i, gap in enumerate(overfit):
                    if gap > 0 and i > 0:
                        ax_zoom.axvspan(val_steps[i-1], val_steps[i],
                                        alpha=0.06, color="red", zorder=0)

        _style(ax_zoom, "Train vs Val (gap = overfitting)", "step", "loss")
        ax_zoom.legend(fontsize=8, framealpha=0.6)

    # ── save ─────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curve → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot MiniFlow training curves")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-id", default=None,
                       help="MiniFlow run ID to plot (default: latest)")
    group.add_argument("--best", action="store_true",
                       help="Plot the run with the lowest val_loss")
    group.add_argument("--all", action="store_true",
                       help="Plot every run, saved as plots/training_curve_<run_id>.png")
    parser.add_argument("--out", default="plots/training_curve.png",
                        help="Output path (default: plots/training_curve.png)")
    args = parser.parse_args()

    from miniflow import ExperimentTracker
    tracker = ExperimentTracker("llm_shakespeare")

    if args.all:
        runs = tracker.get_runs()
        if not runs:
            sys.exit("No MiniFlow runs found.")
        
        plotted = 0
        skipped = 0
        for run in runs:
            metrics = run["metrics"]
            
            # skip runs with no useful data
            if not metrics.get("train_loss") and not metrics.get("val_loss"):
                skipped += 1
                continue
            
            run_id = run["run_id"]
            out = f"plots/training_curve_{run_id}.png"
            print(f"Plotting run: {run_id}")
            plot_curves(run_id, metrics, out)
            plotted += 1
        
        print(f"\nDone: {plotted} plotted, {skipped} skipped (no data)")
    else:
        run_id, metrics = load_run(tracker, args.run_id, args.best)
        print(f"Plotting run: {run_id}")
        plot_curves(run_id, metrics, args.out)


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()