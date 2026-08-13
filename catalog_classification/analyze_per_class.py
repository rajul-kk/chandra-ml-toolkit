"""Follow-up check: aggregate macro-F1 is flat across strategies (see
README) - does that hide a real effect on the rare COMPACT_OBJECT class
specifically, since it's ~3% of the pool and could be swamped by
AGN/STAR in the macro average's... no, macro-F1 already weights classes
equally, so this checks a different thing: whether the *strategies*
differ in how fast they raise COMPACT_OBJECT F1 even if AGN/STAR are
already saturated and dominate the timing of macro-F1 convergence.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    df = pd.read_csv(RESULTS_DIR / "label_efficiency_log.csv")
    class_cols = [c for c in df.columns if c.startswith("f1_")]
    print("per-class F1 columns found:", class_cols)
    if not class_cols:
        print("no per-class columns in this CSV - rerun the experiment first")
        return

    strategies = sorted(df["strategy"].unique())

    for col in class_cols:
        print(f"\n=== {col} ===")
        piv = df.groupby(["strategy", "n_labels"])[col].mean().reset_index()
        for s in strategies:
            sub = piv[piv.strategy == s].sort_values("n_labels")
            early = sub[sub.n_labels <= 150]
            print(f"  {s:22s} at n=30: {sub.iloc[0][col]:.3f}  "
                  f"at n=150: {sub[sub.n_labels==150][col].values[0]:.3f}  "
                  f"plateau(last5): {sub[col].tail(5).mean():.3f}")

        # significance at plateau region (last 5 label-budgets)
        last_ns = sorted(df.n_labels.unique())[-5:]
        rand_vals = df[(df.strategy == "random") & (df.n_labels.isin(last_ns))][col].values
        for s in strategies:
            if s == "random":
                continue
            cand_vals = df[(df.strategy == s) & (df.n_labels.isin(last_ns))][col].values
            t, p = stats.ttest_ind(cand_vals, rand_vals)
            print(f"    {s:22s} plateau mean={cand_vals.mean():.4f} vs random={rand_vals.mean():.4f}  p={p:.4f}")


if __name__ == "__main__":
    main()
