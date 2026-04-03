"""Generate comparison figures across all algorithm variants.

Supports both 3-way (PPO vs DG vs Soft-PPO) and 6-way comparisons
(each algorithm with and without adversarial augmentation).

Usage (6-way comparison):
    python -m rl_captcha.scripts.plot_comparison \
        --logs ppo_noaug=logs/ppo_noaug_training.log \
               ppo_advaug=logs/ppo_advaug_training.log \
               dg_noaug=logs/dg_noaug_training.log \
               dg_advaug=logs/dg_advaug_training.log \
               soft_ppo_noaug=logs/soft_ppo_noaug_training.log \
               soft_ppo_advaug=logs/soft_ppo_advaug_training.log \
        --evals ppo_noaug=logs/ppo_noaug_eval.log \
                ppo_advaug=logs/ppo_advaug_eval.log \
                dg_noaug=logs/dg_noaug_eval.log \
                dg_advaug=logs/dg_advaug_eval.log \
                soft_ppo_noaug=logs/soft_ppo_noaug_eval.log \
                soft_ppo_advaug=logs/soft_ppo_advaug_eval.log \
        --out figures/comparison/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from rl_captcha.scripts.plot_training import parse_log as parse_train_log, smooth
from rl_captcha.scripts.plot_eval import parse_log as parse_eval_log


COLORS = {
    "ppo": "#4a90e2",
    "dg": "#e67e22",
    "soft_ppo": "#2ecc71",
    "ppo_noaug": "#4a90e2",
    "dg_noaug": "#e67e22",
    "soft_ppo_noaug": "#2ecc71",
    "ppo_advaug": "#2a70c2",
    "dg_advaug": "#c66e12",
    "soft_ppo_advaug": "#1eac61",
}
LABELS = {
    "ppo": "PPO",
    "dg": "DG",
    "soft_ppo": "Soft PPO",
    "ppo_noaug": "PPO (no aug)",
    "dg_noaug": "DG (no aug)",
    "soft_ppo_noaug": "Soft PPO (no aug)",
    "ppo_advaug": "PPO (adv aug)",
    "dg_advaug": "DG (adv aug)",
    "soft_ppo_advaug": "Soft PPO (adv aug)",
}
# Line styles: solid for no-aug, dashed for adv-aug
LINESTYLES = {
    "ppo": "-", "dg": "-", "soft_ppo": "-",
    "ppo_noaug": "-", "dg_noaug": "-", "soft_ppo_noaug": "-",
    "ppo_advaug": "--", "dg_advaug": "--", "soft_ppo_advaug": "--",
}


def _parse_kv_args(args: list[str]) -> dict[str, str]:
    """Parse name=path pairs from CLI args."""
    result = {}
    for arg in args:
        if "=" in arg:
            name, path = arg.split("=", 1)
            result[name.strip()] = path.strip()
        else:
            # Infer name from filename
            p = Path(arg)
            name = p.stem.replace("_training", "").replace("_eval", "")
            result[name] = arg
    return result


def plot_comparison(
    all_rollouts: dict[str, list[dict]],
    all_evals: dict[str, dict] | None,
    out_dir: Path,
    fmt: str = "png",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    algos = list(all_rollouts.keys())

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

    # Precompute steps arrays
    steps_k = {}
    for algo in algos:
        steps_k[algo] = np.array([r["steps"] for r in all_rollouts[algo]]) / 1000

    # ── 1. Reward comparison ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo in algos:
        rewards = np.array([r.get("avg_reward", 0) for r in all_rollouts[algo]])
        color = COLORS.get(algo, "#999999")
        label = LABELS.get(algo, algo)
        ls = LINESTYLES.get(algo, "-")
        ax.plot(steps_k[algo], smooth(rewards, 10), color=color, linewidth=2, label=label, linestyle=ls)
        ax.fill_between(steps_k[algo], smooth(rewards, 20) - 0.05,
                        smooth(rewards, 20) + 0.05, color=color, alpha=0.1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Training Steps (x1K)")
    ax.set_ylabel("Average Episode Reward")
    ax.set_title("Training Reward Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"cmp_reward.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_reward.{fmt}")

    # ── 2. Training accuracy comparison ──────────────────────────────
    def _correct_pcts(rollouts):
        arr = []
        for r in rollouts:
            oc = r.get("outcomes", {})
            arr.append(oc.get("correct_allow", 0) + oc.get("correct_block", 0)
                       + oc.get("bot_blocked_puzzle", 0))
        return np.array(arr)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo in algos:
        acc = _correct_pcts(all_rollouts[algo])
        color = COLORS.get(algo, "#999999")
        label = LABELS.get(algo, algo)
        ls = LINESTYLES.get(algo, "-")
        ax.plot(steps_k[algo], smooth(acc, 10), color=color, linewidth=2, label=f"{label} (train)", linestyle=ls)

        # Overlay val accuracy points
        vs, va = [], []
        for r in all_rollouts[algo]:
            if "val_accuracy" in r:
                vs.append(r["steps"] / 1000)
                va.append(r["val_accuracy"] * 100)
        if vs:
            ax.plot(vs, va, "o--", color=color, linewidth=1.2, markersize=3,
                    alpha=0.7, label=f"{label} (val)")

    ax.set_xlabel("Training Steps (x1K)")
    ax.set_ylabel("Correct Decisions (%)")
    ax.set_title("Classification Accuracy Comparison")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"cmp_accuracy.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_accuracy.{fmt}")

    # ── 3. Entropy comparison ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo in algos:
        ent = np.array([r.get("entropy", 0) for r in all_rollouts[algo]])
        color = COLORS.get(algo, "#999999")
        label = LABELS.get(algo, algo)
        ls = LINESTYLES.get(algo, "-")
        ax.plot(steps_k[algo], smooth(ent, 10), color=color, linewidth=2, label=label, linestyle=ls)
    ax.set_xlabel("Training Steps (x1K)")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Decision Confidence (Entropy)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"cmp_entropy.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_entropy.{fmt}")

    # ── 4. Policy Loss comparison ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo in algos:
        ploss = np.array([r.get("policy_loss", 0) for r in all_rollouts[algo]])
        color = COLORS.get(algo, "#999999")
        label = LABELS.get(algo, algo)
        ls = LINESTYLES.get(algo, "-")
        ax.plot(steps_k[algo], smooth(ploss, 10), color=color, linewidth=2, label=label, linestyle=ls)
    ax.set_xlabel("Training Steps (x1K)")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Policy Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"cmp_policy_loss.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_policy_loss.{fmt}")

    # ── 5. Eval metrics bar chart ────────────────────────────────────
    if all_evals:
        eval_algos = [a for a in algos if a in all_evals]
        if eval_algos:
            metric_names = ["Accuracy", "Precision", "Recall", "F1"]
            x = np.arange(len(metric_names))
            n_algos = len(eval_algos)
            width = 0.8 / n_algos

            fig, ax = plt.subplots(figsize=(9, 5))
            for i, algo in enumerate(eval_algos):
                vals = [all_evals[algo].get(m.lower(), 0) for m in metric_names]
                color = COLORS.get(algo, "#999999")
                label = LABELS.get(algo, algo)
                offset = (i - (n_algos - 1) / 2) * width
                bars = ax.bar(x + offset, vals, width, label=label, color=color,
                              edgecolor="white", linewidth=1.5)
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{bar.get_height():.3f}", ha="center", va="bottom",
                            fontsize=9, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(metric_names)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Score")
            ax.set_title("Test Set Evaluation Metrics")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            fig.savefig(out_dir / f"cmp_eval_metrics.{fmt}")
            plt.close(fig)
            print(f"  Saved cmp_eval_metrics.{fmt}")

            # ── 6. Confusion matrices side by side ───────────────────
            n_eval = len(eval_algos)
            fig, axes = plt.subplots(1, n_eval, figsize=(5 * n_eval, 4.5))
            if n_eval == 1:
                axes = [axes]

            cmaps = {
                "ppo": "Blues", "dg": "Oranges", "soft_ppo": "Greens",
                "ppo_noaug": "Blues", "dg_noaug": "Oranges", "soft_ppo_noaug": "Greens",
                "ppo_advaug": "Blues", "dg_advaug": "Oranges", "soft_ppo_advaug": "Greens",
            }
            for ax, algo in zip(axes, eval_algos):
                result = all_evals[algo]
                tp = result.get("tp", 0)
                tn = result.get("tn", 0)
                fp = result.get("fp", 0)
                fn = result.get("fn", 0)
                total = tp + tn + fp + fn or 1
                cm = np.array([[tn, fp], [fn, tp]])
                cm_pct = cm / total * 100

                cmap = cmaps.get(algo, "Greys")
                im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=cm_pct.max() * 1.2)
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Pred Human", "Pred Bot"])
                ax.set_yticklabels(["True Human", "True Bot"])
                for i in range(2):
                    for j in range(2):
                        val = cm[i, j]
                        pct = cm_pct[i, j]
                        color = "white" if pct > cm_pct.max() * 0.6 else "black"
                        ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                                fontsize=13, fontweight="bold", color=color)
                acc = result.get("accuracy", 0)
                f1 = result.get("f1", 0)
                label = LABELS.get(algo, algo)
                ax.set_title(f"{label} (Acc={acc:.3f}, F1={f1:.3f})")

            fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.savefig(out_dir / f"cmp_confusion.{fmt}")
            plt.close(fig)
            print(f"  Saved cmp_confusion.{fmt}")

    # ── 7. Combined summary (2×2 training-only) ─────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Algorithm Training Comparison",
                 fontsize=16, fontweight="bold", y=0.98)

    # (a) Reward
    ax = axes[0, 0]
    for algo in algos:
        rewards = np.array([r.get("avg_reward", 0) for r in all_rollouts[algo]])
        ax.plot(steps_k[algo], smooth(rewards, 10), color=COLORS.get(algo),
                linewidth=2, label=LABELS.get(algo, algo),
                linestyle=LINESTYLES.get(algo, "-"))
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Steps (x1K)")
    ax.set_ylabel("Avg Reward")
    ax.set_title("(a) Training Reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Accuracy
    ax = axes[0, 1]
    for algo in algos:
        acc = _correct_pcts(all_rollouts[algo])
        ax.plot(steps_k[algo], smooth(acc, 10), color=COLORS.get(algo),
                linewidth=2, label=LABELS.get(algo, algo),
                linestyle=LINESTYLES.get(algo, "-"))
    ax.set_xlabel("Steps (x1K)")
    ax.set_ylabel("Correct (%)")
    ax.set_title("(b) Train Accuracy")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Policy Loss
    ax = axes[1, 0]
    for algo in algos:
        ploss = np.array([r.get("policy_loss", 0) for r in all_rollouts[algo]])
        ax.plot(steps_k[algo], smooth(ploss, 10), color=COLORS.get(algo),
                linewidth=2, label=LABELS.get(algo, algo),
                linestyle=LINESTYLES.get(algo, "-"))
    ax.set_xlabel("Steps (x1K)")
    ax.set_ylabel("Policy Loss")
    ax.set_title("(c) Policy Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Entropy
    ax = axes[1, 1]
    for algo in algos:
        ent = np.array([r.get("entropy", 0) for r in all_rollouts[algo]])
        ax.plot(steps_k[algo], smooth(ent, 10), color=COLORS.get(algo),
                linewidth=2, label=LABELS.get(algo, algo),
                linestyle=LINESTYLES.get(algo, "-"))
    ax.set_xlabel("Steps (x1K)")
    ax.set_ylabel("Entropy")
    ax.set_title("(d) Policy Entropy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"cmp_summary.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_summary.{fmt}")

    print(f"\nDone! Comparison figures saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Algorithm comparison figures (with/without adversarial augmentation)")
    parser.add_argument("--logs", type=str, nargs="+", required=True,
                        help="Training logs as name=path pairs (e.g. ppo=logs/ppo_training.log)")
    parser.add_argument("--evals", type=str, nargs="+", default=None,
                        help="Eval logs as name=path pairs (optional)")
    parser.add_argument("--out", type=str, default="figures/comparison", help="Output directory")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    log_map = _parse_kv_args(args.logs)
    for name, path in log_map.items():
        if not Path(path).exists():
            print(f"Error: {name} training log not found: {path}")
            return

    print("Parsing training logs...")
    all_rollouts = {}
    for name, path in log_map.items():
        rollouts = parse_train_log(path)
        if not rollouts:
            print(f"Warning: No rollout data found in {path}")
            continue
        all_rollouts[name] = rollouts
        print(f"  {name}: {len(rollouts)} rollouts")

    if not all_rollouts:
        print("Error: No valid training logs parsed.")
        return

    all_evals = None
    if args.evals:
        eval_map = _parse_kv_args(args.evals)
        all_evals = {}
        for name, path in eval_map.items():
            if Path(path).exists():
                parsed = parse_eval_log(path)
                if parsed:
                    all_evals[name] = parsed
                    print(f"  {name} eval: loaded")

    plot_comparison(all_rollouts, all_evals, Path(args.out), args.format)


if __name__ == "__main__":
    main()
