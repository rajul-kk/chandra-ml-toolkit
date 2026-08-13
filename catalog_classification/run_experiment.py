"""Layer 1 (application) + layer 2 (method) experiment: does active
learning save labels on CSC counterpart-class classification, and does
weighting acquisition by NWAY match reliability recover savings that
vanilla uncertainty sampling leaves on the table?

Usage: python -m catalog_classification.run_experiment [--quick]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

# LightGBM's sklearn wrapper sets feature_names_in_ from auto-generated
# column names at fit time even for plain ndarray input; sklearn then warns
# on predict() with a bare ndarray. Cosmetic only - values are unaffected.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from catalog_classification.dataset import load_feature_pool
from catalog_classification.features import LABELS, build_feature_matrix
from common.active_learning import (
    ActiveLearner,
    class_balanced_uncertainty_score,
    margin_score,
    random_score,
    reliability_weighted,
    uncertainty_score,
)
from common.eval_utils import average_histories, label_savings_fraction, write_csv

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def make_estimator():
    # class_weight='balanced' matters a lot once the labeled set becomes
    # imbalanced (which it does under any strategy, including AL, since
    # the pool itself is 97% AGN/STAR) - checked directly: at a realistic
    # n=830 imbalanced labeled set, this alone lifts COMPACT_OBJECT F1
    # from 0.11 to 0.27 and macro-F1 from 0.63 to 0.69, without hurting
    # AGN/STAR. It's a no-op on the perfectly-balanced stratified init
    # seed, which is why the earlier per-class diagnosis didn't catch it.
    return LGBMClassifier(n_estimators=100, num_leaves=15, min_child_samples=5,
                           verbosity=-1, class_weight="balanced")


def make_eval_fn(estimator_factory, X_test, y_test):
    def eval_fn(model):
        pred = model.predict(X_test)
        per_class = f1_score(y_test, pred, labels=list(range(len(LABELS))), average=None, zero_division=0)
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
        }
        for c_idx, c_name in enumerate(LABELS):
            metrics[f"f1_{c_name}"] = per_class[c_idx]
        return metrics
    return eval_fn


def stratified_indices(y, n_per_class, rng):
    idx = []
    for c in np.unique(y):
        class_idx = np.where(y == c)[0]
        take = min(n_per_class, len(class_idx))
        idx.extend(rng.choice(class_idx, size=take, replace=False))
    return np.array(idx)


def run(n_seeds: int, n_rounds: int, batch_size: int, init_per_class: int, quiet: bool = False):
    pool_df = load_feature_pool()
    X, y, feature_names, reliability = build_feature_matrix(pool_df)

    X_pool_all, X_test, y_pool_all, y_test, rel_pool, _ = train_test_split(
        X, y, reliability, test_size=0.2, stratify=y, random_state=0,
    )

    strategies = {
        "random": lambda: random_score,
        "uncertainty": lambda: uncertainty_score,
        "margin": lambda: margin_score,
        "reliability_weighted": lambda: reliability_weighted(uncertainty_score, rel_pool, alpha=1.0),
        "class_balanced": lambda: class_balanced_uncertainty_score,
    }

    all_histories = {name: [] for name in strategies}
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        init = stratified_indices(y_pool_all, init_per_class, rng)
        eval_fn = make_eval_fn(make_estimator, X_test, y_test)

        for name, score_fn_factory in strategies.items():
            learner = ActiveLearner(
                estimator=make_estimator(),
                X=X_pool_all, label_fn=lambda idx: y_pool_all[idx],
                score_fn=score_fn_factory(), init_indices=init, batch_size=batch_size,
                eval_fn=eval_fn, strategy_name=name, random_state=seed,
            )
            history = learner.run(n_rounds=n_rounds)
            all_histories[name].append(history)
            if not quiet:
                final = history.metrics[-1]
                print(f"seed={seed} strategy={name:22s} "
                      f"final_acc={final['accuracy']:.3f} final_f1={final['macro_f1']:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    flat = [h for hs in all_histories.values() for h in hs]
    write_csv(flat, RESULTS_DIR / "label_efficiency_log.csv")

    averaged = {name: average_histories(hs, strategy_name=name) for name, hs in all_histories.items()}

    print("\n--- label savings vs. random (macro_f1) ---")
    for name in strategies:
        if name == "random":
            continue
        saving = label_savings_fraction(averaged[name], averaged["random"], "macro_f1")
        if saving is None:
            print(f"{name:22s} never matched random's final macro_f1 within budget")
        else:
            print(f"{name:22s} {saving * 100:.1f}% label saving vs random")

    return averaged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast sanity run")
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=40)
    args = parser.parse_args()
    if args.quick:
        run(n_seeds=2, n_rounds=10, batch_size=20, init_per_class=5)
    else:
        run(n_seeds=args.seeds, n_rounds=args.rounds, batch_size=20, init_per_class=10)
