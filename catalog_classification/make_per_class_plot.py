"""Small-multiples plot: per-class F1 vs. labels used, one panel per class,
showing where AL's benefit actually shows up (AGN/STAR) and where it
doesn't (COMPACT_OBJECT) - the finding macro-F1 alone hides.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from catalog_classification.features import LABELS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COLORS = {"random": "tab:blue", "uncertainty": "tab:orange",
          "margin": "tab:green", "reliability_weighted": "tab:red",
          "class_balanced": "tab:purple"}


def main():
    df = pd.read_csv(RESULTS_DIR / "label_efficiency_log.csv")
    strategies = sorted(df["strategy"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

    for ax, label in zip(axes, LABELS):
        col = f"f1_{label}"
        for strat in strategies:
            sub = (df[df.strategy == strat].groupby("n_labels")[col]
                   .agg(["mean", "std"]).reset_index().sort_values("n_labels"))
            ax.plot(sub["n_labels"], sub["mean"], marker="o", markersize=3,
                     label=strat, color=COLORS.get(strat))
            ax.fill_between(sub["n_labels"], sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                             alpha=0.12, color=COLORS.get(strat))
        ax.set_title(label)
        ax.set_xlabel("labels used")
    axes[0].set_ylabel("F1")
    axes[0].legend(fontsize=8)
    fig.suptitle("Per-class F1 vs. labels used (15 seeds): AL helps AGN/STAR, not COMPACT_OBJECT")
    fig.tight_layout()
    out = RESULTS_DIR / "per_class_f1_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
