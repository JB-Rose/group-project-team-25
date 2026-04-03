"""Parse evaluate_ppo.py output and generate evaluation figures.

Supports both single-agent and multi-agent evaluation logs.
When multiple agents are detected, generates per-agent plots plus
side-by-side comparison figures including per-tier and per-family breakdowns.

Usage:
    # Run eval with Tee-Object to capture log
    python -m rl_captcha.scripts.evaluate_ppo \
        --agent ppo=rl_captcha/agent/checkpoints/ppo_run1 \
               dg=rl_captcha/agent/checkpoints/dg_run1 \
               soft_ppo=rl_captcha/agent/checkpoints/soft_ppo_run1 \
        --episodes 500 --split test \
        2>&1 | Tee-Object -FilePath logs/eval_all.log

    # Plot
    python -m rl_captcha.scripts.plot_eval --log logs/eval_all.log --out figures/eval
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Regex patterns matching evaluate_ppo.py output ─────────────────────

RE_AGENT_HEADER = re.compile(r"Loading agent:\s*(\S+)\s+\(")
RE_AGENT_SECTION = re.compile(r"===\s*(\S+)\s*[-\u2014]\s*(\w+)\s+split\s+\((\d+)\s+episodes\)")
RE_SPLIT = re.compile(r"Evaluating on (\w+) split:\s*(\d+) sessions \((\d+) human, (\d+) bot\)")

RE_ACCURACY = re.compile(r"Accuracy:\s*([\d.]+)")
RE_PRECISION = re.compile(r"Precision:\s*([\d.]+)")
RE_RECALL = re.compile(r"Recall:\s*([\d.]+)")
RE_F1 = re.compile(r"F1:\s*([\d.]+)")

RE_TP = re.compile(r"True Positives.*?:\s*(\d+)")
RE_TN = re.compile(r"True Negatives.*?:\s*(\d+)")
RE_FP = re.compile(r"False Positives.*?:\s*(\d+)")
RE_FN = re.compile(r"False Negatives.*?:\s*(\d+)")
RE_TRUNC = re.compile(r"Truncated.*?:\s*(\d+)")

RE_AVG_REWARD = re.compile(r"Avg reward:\s*([\d.+-]+)")
RE_ACTION_LINE = re.compile(r"^\s+(\w[\w_]+)\s+(\d+)\s+\(([\d.]+)%\)")
RE_OUTCOME_LINE = re.compile(r"^\s+([\w_]+)\s+(\d+)\s+\(([\d.]+)%\)")

RE_HUMAN_STEPS = re.compile(r"Avg steps \(human sessions\):\s*([\d.]+)")
RE_BOT_STEPS = re.compile(r"Avg steps \(bot sessions\):\s*([\d.]+)")

# Per-family line: "  stealth             2    25       23     2   92.0%"
RE_FAMILY_LINE = re.compile(
    r"^\s+([\w_]+)\s+(\d+|\?)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%"
)
# Per-tier line: "  Tier 2 (Careful Automation):   45 bots, 88.9% detected"
RE_TIER_LINE = re.compile(
    r"^\s+Tier\s+(\d+)\s+\(([^)]+)\):\s+(\d+)\s+bots?,\s+([\d.]+)%\s+detected"
)

# Consistent colors for algorithms (solid = no aug, dashed-pattern = adv aug)
ALGO_COLORS = {
    "ppo": "#3498db",
    "dg": "#e67e22",
    "soft_ppo": "#9b59b6",
    "ppo_noaug": "#3498db",
    "dg_noaug": "#e67e22",
    "soft_ppo_noaug": "#9b59b6",
    "ppo_advaug": "#1a6fb5",
    "dg_advaug": "#c0641a",
    "soft_ppo_advaug": "#7d3f96",
}
ALGO_LABELS = {
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

# Tier colors (consistent across all plots)
TIER_COLORS = {
    1: "#2ecc71",   # green — commodity (easy)
    2: "#3498db",   # blue — careful automation
    3: "#f39c12",   # orange — semi-automated
    4: "#e74c3c",   # red — trace-conditioned
    5: "#8e44ad",   # purple — LLM-powered
}
TIER_NAMES = {
    1: "T1: Commodity",
    2: "T2: Careful Automation",
    3: "T3: Semi-Automated",
    4: "T4: Trace-Conditioned",
    5: "T5: LLM-Powered",
}


def _detect_encoding(path: str) -> str:
    with open(path, "rb") as fb:
        bom = fb.read(2)
    if bom == b"\xff\xfe":
        return "utf-16-le"
    elif bom == b"\xfe\xff":
        return "utf-16-be"
    return "utf-8"


def parse_log(path: str) -> dict[str, dict]:
    """Parse evaluation log into per-agent result dicts.

    Returns:
        {"agent_name": {metrics...}, ...}
        Also stores global "split" info in a special "_meta" key.
    """
    encoding = _detect_encoding(path)
    agents: dict[str, dict] = {}
    meta: dict = {}
    current_agent: str | None = None
    current: dict = {}
    in_actions = False
    in_outcomes = False
    in_family = False
    in_tier = False

    def _finalize():
        nonlocal current_agent, current
        if current_agent and current:
            current.setdefault("actions", {})
            current.setdefault("outcomes", {})
            current.setdefault("families", {})
            current.setdefault("tiers", {})
            agents[current_agent] = current
        current_agent = None
        current = {}

    with open(path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            raw = line.rstrip()

            # Global split info
            m = RE_SPLIT.search(raw)
            if m:
                meta["split"] = m.group(1)
                meta["total_sessions"] = int(m.group(2))
                meta["human_sessions"] = int(m.group(3))
                meta["bot_sessions"] = int(m.group(4))

            # New agent section: "=== PPO — TEST split (500 episodes) ==="
            m = RE_AGENT_SECTION.search(raw)
            if m:
                _finalize()
                current_agent = m.group(1).lower()
                current = {"split": m.group(2), "episodes": int(m.group(3))}
                in_actions = in_outcomes = in_family = in_tier = False
                continue

            # Also detect "Loading agent: ppo (...)"
            m = RE_AGENT_HEADER.search(raw)
            if m and not current_agent:
                _finalize()
                current_agent = m.group(1).lower()
                current = {}
                in_actions = in_outcomes = in_family = in_tier = False
                continue

            if not current_agent:
                continue

            # Scalar metrics
            for name, regex in [("accuracy", RE_ACCURACY), ("precision", RE_PRECISION),
                                ("recall", RE_RECALL), ("f1", RE_F1)]:
                m = regex.search(raw)
                if m:
                    current[name] = float(m.group(1))

            for name, regex in [("tp", RE_TP), ("tn", RE_TN), ("fp", RE_FP),
                                ("fn", RE_FN), ("truncated", RE_TRUNC)]:
                m = regex.search(raw)
                if m:
                    current[name] = int(m.group(1))

            m = RE_AVG_REWARD.search(raw)
            if m:
                current["avg_reward"] = float(m.group(1))

            m = RE_HUMAN_STEPS.search(raw)
            if m:
                current["human_avg_steps"] = float(m.group(1))
            m = RE_BOT_STEPS.search(raw)
            if m:
                current["bot_avg_steps"] = float(m.group(1))

            # Section toggles
            if "Per-Family Bot Detection" in raw:
                in_family = True
                in_tier = in_actions = in_outcomes = False
                current.setdefault("families", {})
                continue
            if "Per-Tier Summary" in raw:
                in_tier = True
                in_family = in_actions = in_outcomes = False
                current.setdefault("tiers", {})
                continue
            if "Final Action Distribution" in raw:
                in_actions = True
                in_family = in_tier = in_outcomes = False
                current.setdefault("actions", {})
                continue
            if "Outcome Distribution" in raw:
                in_outcomes = True
                in_family = in_tier = in_actions = False
                current.setdefault("outcomes", {})
                continue
            if raw.strip().startswith("---") and ("Confusion" in raw or not in_actions):
                if not in_actions and not in_outcomes:
                    pass
                else:
                    in_actions = in_outcomes = False
                    continue

            if in_family:
                m = RE_FAMILY_LINE.search(raw)
                if m:
                    family = m.group(1)
                    tier = int(m.group(2)) if m.group(2) != "?" else 0
                    current.setdefault("families", {})[family] = {
                        "tier": tier,
                        "n": int(m.group(3)),
                        "detected": int(m.group(4)),
                        "missed": int(m.group(5)),
                        "rate": float(m.group(6)) / 100.0,
                    }

            if in_tier:
                m = RE_TIER_LINE.search(raw)
                if m:
                    tier_num = int(m.group(1))
                    current.setdefault("tiers", {})[tier_num] = {
                        "name": m.group(2),
                        "n": int(m.group(3)),
                        "rate": float(m.group(4)) / 100.0,
                    }

            if in_actions:
                m = RE_ACTION_LINE.search(raw)
                if m:
                    current.setdefault("actions", {})[m.group(1)] = int(m.group(2))

            if in_outcomes:
                m = RE_OUTCOME_LINE.search(raw)
                if m:
                    current.setdefault("outcomes", {})[m.group(1)] = int(m.group(2))

    _finalize()
    agents["_meta"] = meta
    return agents


def _get_color(name: str) -> str:
    return ALGO_COLORS.get(name.lower(), "#34495e")


def _get_label(name: str) -> str:
    return ALGO_LABELS.get(name.lower(), name.upper())


def _setup_style():
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


def plot_single(name: str, result: dict, out_dir: Path, fmt: str = "png"):
    """Generate per-agent evaluation plots."""
    _setup_style()
    split_name = result.get("split", "test").upper()
    label = _get_label(name)

    # Confusion matrix
    tp = result.get("tp", 0)
    tn = result.get("tn", 0)
    fp = result.get("fp", 0)
    fn = result.get("fn", 0)
    total = tp + tn + fp + fn or 1

    cm = np.array([[tn, fp], [fn, tp]])
    cm_pct = cm / total * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=cm_pct.max() * 1.2)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nHuman", "Predicted\nBot"])
    ax.set_yticklabels(["Actual\nHuman", "Actual\nBot"])
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = cm_pct[i, j]
            color = "white" if pct > cm_pct.max() * 0.6 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)
    ax.set_title(f"{label} — Confusion Matrix ({split_name})")
    fig.colorbar(im, ax=ax, label="% of episodes", shrink=0.8)
    fig.savefig(out_dir / f"eval_{name}_confusion.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_{name}_confusion.{fmt}")

    # Action distribution
    actions = result.get("actions", {})
    if actions:
        action_colors = {
            "allow": "#2ecc71", "block": "#e74c3c",
            "easy_puzzle": "#f1c40f", "medium_puzzle": "#e67e22", "hard_puzzle": "#d35400",
            "continue": "#95a5a6", "deploy_honeypot": "#3498db",
        }
        fig, ax = plt.subplots(figsize=(7, 4))
        action_names = list(actions.keys())
        action_counts = [actions[a] for a in action_names]
        colors = [action_colors.get(a, "#bdc3c7") for a in action_names]
        bars = ax.barh(action_names, action_counts, color=colors, edgecolor="white", linewidth=1.2)
        total_actions = sum(action_counts)
        for bar, count in zip(bars, action_counts):
            pct = count / total_actions * 100 if total_actions else 0
            ax.text(bar.get_width() + total_actions * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{count} ({pct:.1f}%)", va="center", fontsize=10)
        ax.set_xlabel("Count")
        ax.set_title(f"{label} — Final Action Distribution ({split_name})")
        ax.grid(True, axis="x", alpha=0.3)
        fig.savefig(out_dir / f"eval_{name}_actions.{fmt}")
        plt.close(fig)
        print(f"  Saved eval_{name}_actions.{fmt}")

    # Per-family detection rate (single agent)
    families = result.get("families", {})
    if families:
        _plot_family_bars(name, families, out_dir, fmt, split_name)

    # Per-tier detection rate (single agent)
    tiers = result.get("tiers", {})
    if tiers:
        _plot_tier_bars_single(name, tiers, out_dir, fmt, split_name)


def _plot_family_bars(name: str, families: dict, out_dir: Path, fmt: str, split_name: str):
    """Per-family horizontal bar chart with detection rates, colored by tier."""
    label = _get_label(name)
    # Sort by tier then name
    sorted_fams = sorted(families.items(), key=lambda x: (x[1]["tier"], x[0]))

    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_fams) * 0.6 + 1)))

    fam_names = [f for f, _ in sorted_fams]
    rates = [d["rate"] for _, d in sorted_fams]
    counts = [d["n"] for _, d in sorted_fams]
    colors = [TIER_COLORS.get(d["tier"], "#95a5a6") for _, d in sorted_fams]

    y = np.arange(len(fam_names))
    bars = ax.barh(y, rates, color=colors, edgecolor="white", linewidth=1.2, height=0.7)

    for i, (bar, rate, n) in enumerate(zip(bars, rates, counts)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{rate:.0%} (n={n})", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(fam_names)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Detection Rate")
    ax.set_title(f"{label} — Per-Family Bot Detection ({split_name})")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.3)

    # Tier legend
    from matplotlib.patches import Patch
    tier_nums = sorted(set(d["tier"] for _, d in sorted_fams))
    legend_handles = [Patch(color=TIER_COLORS.get(t, "#95a5a6"),
                            label=TIER_NAMES.get(t, f"Tier {t}")) for t in tier_nums]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"eval_{name}_per_family.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_{name}_per_family.{fmt}")


def _plot_tier_bars_single(name: str, tiers: dict, out_dir: Path, fmt: str, split_name: str):
    """Per-tier bar chart for a single agent."""
    label = _get_label(name)

    tier_nums = sorted(tiers.keys())
    rates = [tiers[t]["rate"] for t in tier_nums]
    counts = [tiers[t]["n"] for t in tier_nums]
    colors = [TIER_COLORS.get(t, "#95a5a6") for t in tier_nums]
    labels = [TIER_NAMES.get(t, f"Tier {t}") for t in tier_nums]

    fig, ax = plt.subplots(figsize=(max(6, len(tier_nums) * 1.8 + 1), 5))
    x = np.arange(len(tier_nums))
    bars = ax.bar(x, rates, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

    for bar, rate, n in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.0%}\n(n={n})", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Detection Rate")
    ax.set_title(f"{label} — Per-Tier Detection Rate ({split_name})")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"eval_{name}_per_tier.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_{name}_per_tier.{fmt}")


def plot_tier_comparison(agents: dict[str, dict], out_dir: Path, fmt: str = "png"):
    """Per-tier detection rates compared across all agents (grouped bar chart)."""
    names = [n for n in agents if n != "_meta"]
    # Collect all tier numbers across agents
    all_tiers = set()
    for name in names:
        all_tiers.update(agents[name].get("tiers", {}).keys())
    if not all_tiers:
        return

    _setup_style()
    meta = agents.get("_meta", {})
    split_name = meta.get("split", "test").upper()
    tier_nums = sorted(all_tiers)

    x = np.arange(len(tier_nums))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(max(8, len(tier_nums) * 2.5), 6))

    for i, name in enumerate(names):
        tiers = agents[name].get("tiers", {})
        rates = [tiers.get(t, {}).get("rate", 0) for t in tier_nums]
        offset = (i - len(names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=_get_label(name),
                      color=_get_color(name), edgecolor="white", linewidth=1)
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{rate:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    tier_labels = [TIER_NAMES.get(t, f"Tier {t}") for t in tier_nums]
    # Add sample counts below tier labels
    for name in names:
        tiers = agents[name].get("tiers", {})
        for t_idx, t in enumerate(tier_nums):
            n = tiers.get(t, {}).get("n", 0)
            if n > 0:
                ax.text(x[t_idx], -0.08, f"n={n}", ha="center", va="top",
                        fontsize=8, color="gray", transform=ax.get_xaxis_transform())
                break  # Only show once per tier

    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Detection Rate")
    ax.set_title(f"Per-Tier Detection Rate — Algorithm Comparison ({split_name})")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"eval_tier_comparison.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_tier_comparison.{fmt}")


def plot_family_heatmap(agents: dict[str, dict], out_dir: Path, fmt: str = "png"):
    """Heatmap: agents × bot families, cell = detection rate."""
    names = [n for n in agents if n != "_meta"]
    # Collect all families
    all_families = set()
    for name in names:
        all_families.update(agents[name].get("families", {}).keys())
    if not all_families:
        return

    _setup_style()
    meta = agents.get("_meta", {})
    split_name = meta.get("split", "test").upper()

    # Sort families by tier then name
    def _sort_key(fam):
        for name in names:
            info = agents[name].get("families", {}).get(fam)
            if info:
                return (info["tier"], fam)
        return (99, fam)

    families = sorted(all_families, key=_sort_key)

    # Build matrix
    matrix = np.zeros((len(names), len(families)))
    counts = np.zeros((len(names), len(families)), dtype=int)
    for i, name in enumerate(names):
        fam_data = agents[name].get("families", {})
        for j, fam in enumerate(families):
            info = fam_data.get(fam, {})
            matrix[i, j] = info.get("rate", 0)
            counts[i, j] = info.get("n", 0)

    fig, ax = plt.subplots(figsize=(max(8, len(families) * 1.2 + 2), max(4, len(names) * 1.2 + 1)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(families)):
            rate = matrix[i, j]
            n = counts[i, j]
            color = "white" if rate < 0.4 or rate > 0.85 else "black"
            ax.text(j, i, f"{rate:.0%}\n(n={n})", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(np.arange(len(families)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(families, rotation=35, ha="right")
    ax.set_yticklabels([_get_label(n) for n in names])

    # Add tier separators
    prev_tier = None
    for j, fam in enumerate(families):
        for name in names:
            info = agents[name].get("families", {}).get(fam)
            if info:
                tier = info["tier"]
                if prev_tier is not None and tier != prev_tier:
                    ax.axvline(x=j - 0.5, color="black", linewidth=2, alpha=0.5)
                prev_tier = tier
                break

    ax.set_title(f"Detection Rate by Bot Family × Algorithm ({split_name})")
    fig.colorbar(im, ax=ax, label="Detection Rate", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / f"eval_family_heatmap.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_family_heatmap.{fmt}")


def plot_comparison(agents: dict[str, dict], out_dir: Path, fmt: str = "png"):
    """Generate multi-agent comparison plots."""
    names = [n for n in agents if n != "_meta"]
    if not names:
        return

    _setup_style()
    meta = agents.get("_meta", {})
    split_name = meta.get("split", "test").upper()

    # ── 1. Grouped metrics bar chart ─────────────────────────────────
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(metric_keys))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        r = agents[name]
        values = [r.get(k, 0) for k in metric_keys]
        offset = (i - len(names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=_get_label(name),
                      color=_get_color(name), edgecolor="white", linewidth=1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Evaluation Metrics Comparison — {split_name} Split")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_dir / f"eval_metrics_comparison.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_metrics_comparison.{fmt}")

    # ── 2. Confusion matrices side by side ───────────────────────────
    n_agents = len(names)
    fig, axes = plt.subplots(1, n_agents, figsize=(5 * n_agents, 4))
    if n_agents == 1:
        axes = [axes]

    for idx, name in enumerate(names):
        ax = axes[idx]
        r = agents[name]
        tp = r.get("tp", 0)
        tn = r.get("tn", 0)
        fp = r.get("fp", 0)
        fn = r.get("fn", 0)
        total = tp + tn + fp + fn or 1
        cm = np.array([[tn, fp], [fn, tp]])
        cm_pct = cm / total * 100

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=max(cm_pct.max() * 1.2, 1))
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
                        fontsize=12, fontweight="bold", color=color)
        ax.set_title(_get_label(name))

    fig.suptitle(f"Confusion Matrices — {split_name} Split", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"eval_confusion_comparison.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_confusion_comparison.{fmt}")

    # ── 3. Decision timing comparison ────────────────────────────────
    has_timing = all(
        agents[n].get("human_avg_steps") is not None and agents[n].get("bot_avg_steps") is not None
        for n in names
    )
    if has_timing:
        x = np.arange(2)
        width = 0.8 / len(names)
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, name in enumerate(names):
            r = agents[name]
            values = [r["human_avg_steps"], r["bot_avg_steps"]]
            offset = (i - len(names) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=_get_label(name),
                          color=_get_color(name), edgecolor="white", linewidth=1)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["Human", "Bot"])
        ax.set_ylabel("Avg Windows Before Decision")
        ax.set_title(f"Decision Timing — {split_name} Split")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.savefig(out_dir / f"eval_timing_comparison.{fmt}")
        plt.close(fig)
        print(f"  Saved eval_timing_comparison.{fmt}")

    # ── 4. Per-tier comparison ────────────────────────────────────────
    plot_tier_comparison(agents, out_dir, fmt)

    # ── 5. Family heatmap ─────────────────────────────────────────────
    plot_family_heatmap(agents, out_dir, fmt)

    # ── 6. Combined summary (2×3 grid) ───────────────────────────────
    _plot_combined_summary(agents, names, split_name, out_dir, fmt)


def _plot_combined_summary(agents, names, split_name, out_dir, fmt):
    """2×3 combined summary figure with tier and family data."""
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Evaluation Summary — {split_name} Split", fontsize=16, fontweight="bold", y=0.98)

    # (a) Metrics comparison
    ax = axes[0, 0]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        r = agents[name]
        values = [r.get(k, 0) for k in metric_keys]
        offset = (i - len(names) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=_get_label(name), color=_get_color(name))
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_title("(a) Global Metrics")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # (b) TP/TN/FP/FN stacked
    ax = axes[0, 1]
    categories = ["TP", "TN", "FP", "FN"]
    cat_keys = ["tp", "tn", "fp", "fn"]
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    cat_colors = ["#2ecc71", "#3498db", "#e74c3c", "#e67e22"]
    for ci, (cat, ck) in enumerate(zip(categories, cat_keys)):
        vals = [agents[n].get(ck, 0) for n in names]
        ax.bar(x, vals, 0.6, bottom=bottom, label=cat, color=cat_colors[ci])
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([_get_label(n) for n in names])
    ax.set_title("(b) Outcome Breakdown")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # (c) Per-tier detection rates (grouped)
    ax = axes[0, 2]
    all_tiers = set()
    for name in names:
        all_tiers.update(agents[name].get("tiers", {}).keys())
    if all_tiers:
        tier_nums = sorted(all_tiers)
        x = np.arange(len(tier_nums))
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            tiers = agents[name].get("tiers", {})
            rates = [tiers.get(t, {}).get("rate", 0) for t in tier_nums]
            offset = (i - len(names) / 2 + 0.5) * width
            ax.bar(x + offset, rates, width, label=_get_label(name), color=_get_color(name))
        tier_labels = [TIER_NAMES.get(t, f"T{t}") for t in tier_nums]
        ax.set_xticks(x)
        ax.set_xticklabels(tier_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_title("(c) Per-Tier Detection Rate")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # (d) Action distributions (grouped)
    ax = axes[1, 0]
    all_actions = set()
    for name in names:
        all_actions.update(agents[name].get("actions", {}).keys())
    all_actions = sorted(all_actions)
    if all_actions:
        y = np.arange(len(all_actions))
        bar_h = 0.8 / len(names)
        for i, name in enumerate(names):
            acts = agents[name].get("actions", {})
            vals = [acts.get(a, 0) for a in all_actions]
            offset = (i - len(names) / 2 + 0.5) * bar_h
            ax.barh(y + offset, vals, bar_h, label=_get_label(name), color=_get_color(name))
        ax.set_yticks(y)
        ax.set_yticklabels(all_actions)
    ax.set_xlabel("Count")
    ax.set_title("(d) Final Actions")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    # (e) Per-family detection rate (first agent, as heatmap-lite)
    ax = axes[1, 1]
    first_agent = names[0]
    families = agents[first_agent].get("families", {})
    if families:
        sorted_fams = sorted(families.items(), key=lambda x: (x[1]["tier"], x[0]))
        fam_names = [f for f, _ in sorted_fams]
        rates = [d["rate"] for _, d in sorted_fams]
        counts = [d["n"] for _, d in sorted_fams]
        colors = [TIER_COLORS.get(d["tier"], "#95a5a6") for _, d in sorted_fams]
        y = np.arange(len(fam_names))
        bars = ax.barh(y, rates, color=colors, height=0.7)
        for bar, rate, n in zip(bars, rates, counts):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{rate:.0%} (n={n})", va="center", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(fam_names, fontsize=9)
        ax.set_xlim(0, 1.3)
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_title(f"(e) Per-Family ({_get_label(first_agent)})")
    ax.grid(True, axis="x", alpha=0.3)

    # (f) Decision timing
    ax = axes[1, 2]
    has_timing = all(
        agents[n].get("human_avg_steps") is not None and agents[n].get("bot_avg_steps") is not None
        for n in names
    )
    if has_timing:
        x = np.arange(2)
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            r = agents[name]
            values = [r["human_avg_steps"], r["bot_avg_steps"]]
            offset = (i - len(names) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=_get_label(name), color=_get_color(name))
        ax.set_xticks(x)
        ax.set_xticklabels(["Human", "Bot"])
    ax.set_ylabel("Avg Windows")
    ax.set_title("(f) Decision Timing")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"eval_summary.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_summary.{fmt}")


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--log", type=str, required=True, help="Path to evaluation log file")
    parser.add_argument("--out", type=str, default="figures", help="Output directory for figures")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
                        help="Figure format (pdf recommended for papers)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {log_path}...")
    agents = parse_log(str(log_path))

    agent_names = [n for n in agents if n != "_meta"]
    if not agent_names:
        print("No evaluation data found in log file.")
        return

    print(f"Found {len(agent_names)} agent(s): {', '.join(agent_names)}")
    for name in agent_names:
        r = agents[name]
        print(f"  {name}: accuracy={r.get('accuracy', 'N/A')}, f1={r.get('f1', 'N/A')}")
        tiers = r.get("tiers", {})
        if tiers:
            for t in sorted(tiers):
                print(f"    Tier {t}: {tiers[t]['rate']:.0%} ({tiers[t]['n']} bots)")

    # Per-agent plots
    for name in agent_names:
        plot_single(name, agents[name], out_dir, fmt=args.format)

    # Comparison plots (always generated, even for single agent)
    plot_comparison(agents, out_dir, fmt=args.format)

    print(f"\nDone! Evaluation figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
