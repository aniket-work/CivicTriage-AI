"""Training summary plots for the experimental PoC."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_metric_bars(
    before: dict[str, float],
    after: dict[str, float],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    keys = ["accuracy", "macro_f1"]
    x = range(len(keys))
    before_vals = [before[k] for k in keys]
    after_vals = [after[k] for k in keys]
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], before_vals, width, label="SFT baseline")
    ax.bar([i + width / 2 for i in x], after_vals, width, label="After preference alignment")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Accuracy", "Macro F1"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("CivicTriage-AI: PoC routing metrics (synthetic eval)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_class_distribution(labels: list[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    counts: dict[str, int] = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    names = sorted(counts.keys())
    vals = [counts[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals, color="#2c5282")
    ax.set_title("Synthetic training label distribution")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
