"""Layer 2 diagnostic: does vanilla uncertainty sampling preferentially
query low-reliability (less trustworthy) sources, and is that correlated
with the rare COMPACT_OBJECT class?

This is the direct test of the premise behind reliability_weighted() in
common/active_learning.py - run before trusting any result about whether
reliability weighting helps or hurts.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pointbiserialr, spearmanr
from sklearn.model_selection import train_test_split

from catalog_classification.dataset import load_feature_pool
from catalog_classification.features import LABELS, build_feature_matrix
from common.active_learning import ActiveLearner, uncertainty_score


def main():
    pool_df = load_feature_pool()
    X, y, feature_names, reliability = build_feature_matrix(pool_df)

    X_pool, X_test, y_pool, y_test, rel_pool, _ = train_test_split(
        X, y, reliability, test_size=0.2, stratify=y, random_state=0,
    )

    from lightgbm import LGBMClassifier

    rng = np.random.RandomState(0)
    init = []
    for c in np.unique(y_pool):
        idx = np.where(y_pool == c)[0]
        init.extend(rng.choice(idx, size=min(10, len(idx)), replace=False))
    init = np.array(init)

    learner = ActiveLearner(
        estimator=LGBMClassifier(n_estimators=100, num_leaves=15, min_child_samples=5, verbosity=-1),
        X=X_pool, label_fn=lambda idx: y_pool[idx],
        score_fn=uncertainty_score, init_indices=init, batch_size=20,
        strategy_name="uncertainty", random_state=0,
    )
    model = learner._fit_current()
    all_queried = []
    for _ in range(40):
        queried = learner.step(model)
        if len(queried) == 0:
            break
        all_queried.extend(queried)
        model = learner._fit_current()
    all_queried = np.array(all_queried)

    # Reliability: queried-by-uncertainty vs. never-queried
    queried_mask = np.zeros(len(X_pool), dtype=bool)
    queried_mask[all_queried] = True
    # exclude the stratified init set from "queried by strategy" comparison
    queried_mask[init] = False
    unqueried_mask = ~queried_mask
    unqueried_mask[init] = False

    rel_queried = rel_pool[queried_mask]
    rel_unqueried = rel_pool[unqueried_mask]
    print(f"n queried by uncertainty (excl. init): {len(rel_queried)}")
    print(f"n never queried: {len(rel_unqueried)}")
    print(f"mean p_match_ind, queried:     {rel_queried.mean():.4f}")
    print(f"mean p_match_ind, unqueried:   {rel_unqueried.mean():.4f}")

    r, p = pointbiserialr(queried_mask[~np.isin(np.arange(len(X_pool)), init)].astype(int),
                           rel_pool[~np.isin(np.arange(len(X_pool)), init)])
    print(f"\npoint-biserial correlation(queried, reliability) = {r:.4f} (p={p:.2e})")

    # class breakdown of queried set vs full pool
    print("\nclass share: full pool vs. queried-by-uncertainty")
    for c_idx, c_name in enumerate(LABELS):
        pool_share = (y_pool == c_idx).mean()
        queried_share = (y_pool[all_queried] == c_idx).mean()
        print(f"  {c_name:16s} pool={pool_share:.3f}  queried={queried_share:.3f}")

    # reliability by class, in the pool
    print("\nmean p_match_ind by class (full pool):")
    for c_idx, c_name in enumerate(LABELS):
        print(f"  {c_name:16s} {rel_pool[y_pool == c_idx].mean():.4f}")


if __name__ == "__main__":
    main()
