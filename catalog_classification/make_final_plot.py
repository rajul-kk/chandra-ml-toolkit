"""Regenerate the label-efficiency plot from the saved results CSV
(results/label_efficiency_log.csv), without rerunning the experiment.
"""
from pathlib import Path

import pandas as pd

from common.eval_utils import plot_label_efficiency_curve

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def averaged_from_csv(df: pd.DataFrame, strategy: str) -> dict:
    sub = (df[df.strategy == strategy]
           .groupby("n_labels")
           .agg(f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"))
           .reset_index()
           .sort_values("n_labels"))
    return {
        "n_labels": sub["n_labels"].tolist(),
        "strategy": strategy,
        "macro_f1": {"mean": sub["f1_mean"].tolist(), "std": sub["f1_std"].fillna(0).tolist()},
    }


def main():
    df = pd.read_csv(RESULTS_DIR / "label_efficiency_log.csv")
    strategies = ["random", "uncertainty", "margin", "reliability_weighted"]
    averaged = [averaged_from_csv(df, s) for s in strategies]
    plot_label_efficiency_curve(
        averaged, metric="macro_f1", path=RESULTS_DIR / "label_efficiency_curve.png",
        title="CSC 2.1 counterpart-class classification: macro-F1 vs. labels used (15 seeds)",
    )
    print(f"wrote {RESULTS_DIR / 'label_efficiency_curve.png'}")


if __name__ == "__main__":
    main()
