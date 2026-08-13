"""Shared plotting/metrics for label-efficiency comparisons.

Takes LearningHistory objects from common.active_learning and turns them
into the standard artifact of a label-efficiency study: a curve of a
metric (accuracy, macro-F1, ...) vs. labels used, one line per strategy,
plus the headline number - how many labels a strategy needs to match a
reference strategy's final score.

Nothing here is specific to X-ray classification; any future comparison
in this repo (image anomaly detection, event-file representation
learning) reuses this as-is by handing it its own LearningHistory list.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from common.active_learning import LearningHistory


def history_to_rows(history: LearningHistory) -> List[dict]:
    """Flatten one LearningHistory into CSV-ready rows."""
    rows = []
    for n_labels, metrics in zip(history.n_labels, history.metrics):
        row = {"strategy": history.strategy_name, "n_labels": n_labels}
        row.update(metrics)
        rows.append(row)
    return rows


def write_csv(histories: List[LearningHistory], path: Path) -> None:
    """Log every round of every strategy's history to one CSV, long format
    (one row per strategy per round) so it's trivial to pivot/plot later
    without re-running the experiment.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [row for h in histories for row in history_to_rows(h)]
    if not all_rows:
        raise ValueError("no history rows to write")
    fieldnames = list(dict.fromkeys(k for row in all_rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def average_histories(histories: List[LearningHistory], strategy_name: Optional[str] = None) -> dict:
    """Average metrics across repeated runs of the same strategy (different
    seeds), aligned by n_labels. All histories must share the same
    n_labels schedule (true whenever runs share init_size/batch_size/rounds).

    Returns {"n_labels": [...], metric_name: {"mean": [...], "std": [...]}, ...}
    """
    if not histories:
        raise ValueError("no histories to average")
    n_labels = histories[0].n_labels
    for h in histories:
        if h.n_labels != n_labels:
            raise ValueError("histories have mismatched n_labels schedules; "
                              "use identical init_size/batch_size/rounds across seeds")

    metric_names = sorted(histories[0].metrics[0].keys())
    out = {"n_labels": n_labels, "strategy": strategy_name or histories[0].strategy_name}
    for m in metric_names:
        values = np.array([[round_metrics[m] for round_metrics in h.metrics] for h in histories])
        out[m] = {"mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist()}
    return out


def labels_to_match(candidate: dict, reference: dict, metric: str) -> Optional[int]:
    """How many labels `candidate` needs to first reach `reference`'s final
    score on `metric`. Returns None if it never does within the run.

    Both args are averaged-history dicts (from average_histories) sharing
    the same metric. This is the headline "label savings" number.
    """
    target = reference[metric]["mean"][-1]
    means = candidate[metric]["mean"]
    for n, score in zip(candidate["n_labels"], means):
        if score >= target:
            return n
    return None


def label_savings_fraction(candidate: dict, reference: dict, metric: str) -> Optional[float]:
    """Fraction of reference's final label budget saved by candidate
    reaching the same score, e.g. 0.6 means candidate used 60% fewer labels.
    None if candidate never matched reference within its run.
    """
    n_needed = labels_to_match(candidate, reference, metric)
    if n_needed is None:
        return None
    n_reference = reference["n_labels"][-1]
    return 1.0 - (n_needed / n_reference)


def plot_label_efficiency_curve(averaged: List[dict], metric: str, path: Path,
                                 title: Optional[str] = None) -> None:
    """AL-vs-random-style curve: metric vs. labels used, one line per
    strategy in `averaged` (list of average_histories() outputs), shaded
    +/- 1 std across seeds.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for entry in averaged:
        n = entry["n_labels"]
        mean = np.array(entry[metric]["mean"])
        std = np.array(entry[metric]["std"])
        ax.plot(n, mean, marker="o", label=entry["strategy"])
        ax.fill_between(n, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("labels used")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs. labels used")
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
