#!/usr/bin/env python3
"""
CivicTriage-AI entrypoint: supervised routing fit + preference-style alignment.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from civic_triage.modeling import (  # noqa: E402
    apply_preference_alignment,
    fit_sft,
    metrics_for,
)
from civic_triage.plots import plot_class_distribution, plot_metric_bars  # noqa: E402
from civic_triage.reporting import ascii_table, summarize_run  # noqa: E402
from civic_triage.synthetic import generate_labeled_requests, iter_preference_pairs  # noqa: E402


def run_pipeline(seed: int, n_per_class: int, pref_mistake_rate: float) -> int:
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    data = generate_labeled_requests(n_per_class=n_per_class, seed=seed)
    split = int(len(data) * 0.8)
    train = data[:split]
    test = data[split:]
    train_texts = [x.text for x in train]
    train_labels = [x.label for x in train]
    test_texts = [x.text for x in test]
    test_labels = [x.label for x in test]

    plot_class_distribution(
        train_labels,
        os.path.join(ROOT, "output", "label_distribution.png"),
    )

    sft = fit_sft(train_texts, train_labels, seed=seed)
    sft_metrics = metrics_for(sft, test_texts, test_labels)

    pairs = list(
        iter_preference_pairs(train, mistake_rate=pref_mistake_rate, seed=seed + 1)
    )
    pair_tuples = [(p.text, p.chosen, p.rejected) for p in pairs]
    aug_texts, aug_labels = apply_preference_alignment(
        train_texts,
        train_labels,
        pair_tuples,
        oversample_chosen=3,
        seed=seed + 2,
    )
    aligned = fit_sft(aug_texts, aug_labels, seed=seed + 3)
    aligned_metrics = metrics_for(aligned, test_texts, test_labels)

    plot_metric_bars(
        sft_metrics,
        aligned_metrics,
        os.path.join(ROOT, "output", "metrics_compare.png"),
    )

    print("=" * 72)
    print("  CivicTriage-AI — experimental municipal routing PoC")
    print("=" * 72)
    print()
    print(summarize_run("after_sft", sft_metrics, len(train_texts), 0))
    print()
    print(summarize_run("after_alignment", aligned_metrics, len(aug_texts), len(pairs)))
    print()
    summary_rows = [
        ["after_sft", sft_metrics["accuracy"], sft_metrics["macro_f1"]],
        ["after_alignment", aligned_metrics["accuracy"], aligned_metrics["macro_f1"]],
    ]
    print(
        ascii_table(
            ["phase", "accuracy", "macro_f1"],
            summary_rows,
        )
    )
    print()
    print("Artifacts written to ./output (metrics charts, label distribution).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CivicTriage-AI PoC runner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-class", type=int, default=120)
    parser.add_argument("--pref-noise", type=float, default=0.22)
    args = parser.parse_args()
    return run_pipeline(
        seed=args.seed,
        n_per_class=args.n_per_class,
        pref_mistake_rate=args.pref_noise,
    )


if __name__ == "__main__":
    raise SystemExit(main())
