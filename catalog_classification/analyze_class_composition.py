"""Follow-up check: class_balanced_uncertainty_score didn't move
COMPACT_OBJECT F1 at all (see analyze_per_class.py). Two possible reasons:
(1) it didn't actually query more COMPACT_OBJECT examples than plain
uncertainty sampling did, or (2) it did, and the class is just not
separable with this feature set regardless of which examples get
labeled. This measures which.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from catalog_classification.dataset import load_feature_pool
from catalog_classification.features import LABELS, build_feature_matrix
from common.active_learning import ActiveLearner, class_balanced_uncertainty_score, uncertainty_score


def run_and_count_classes(score_fn, name, X_pool, y_pool, init, n_rounds=40, batch_size=20, seed=0):
    learner = ActiveLearner(
        estimator=LGBMClassifier(n_estimators=100, num_leaves=15, min_child_samples=5,
                                  verbosity=-1, class_weight="balanced"),
        X=X_pool, label_fn=lambda idx: y_pool[idx],
        score_fn=score_fn, init_indices=init, batch_size=batch_size,
        strategy_name=name, random_state=seed,
    )
    model = learner._fit_current()
    all_queried = []
    for _ in range(n_rounds):
        queried = learner.step(model)
        if len(queried) == 0:
            break
        all_queried.extend(queried)
        model = learner._fit_current()
    all_queried = np.array(all_queried)
    counts = {c: int((y_pool[all_queried] == i).sum()) for i, c in enumerate(LABELS)}
    return counts, len(all_queried)


def main():
    pool_df = load_feature_pool()
    X, y, feature_names, reliability = build_feature_matrix(pool_df)
    X_pool, X_test, y_pool, y_test, _, _ = train_test_split(
        X, y, reliability, test_size=0.2, stratify=y, random_state=0,
    )

    rng = np.random.RandomState(0)
    init = []
    for c in np.unique(y_pool):
        idx = np.where(y_pool == c)[0]
        init.extend(rng.choice(idx, size=min(10, len(idx)), replace=False))
    init = np.array(init)

    n_compact_in_pool = int((y_pool == LABELS.index("COMPACT_OBJECT")).sum())
    print(f"COMPACT_OBJECT examples available in pool (excl. init): "
          f"{n_compact_in_pool - (y_pool[init] == LABELS.index('COMPACT_OBJECT')).sum()}")

    for score_fn, name in [(uncertainty_score, "uncertainty"),
                            (class_balanced_uncertainty_score, "class_balanced")]:
        counts, total = run_and_count_classes(score_fn, name, X_pool, y_pool, init)
        print(f"\n{name}: {total} labels queried")
        for c, n in counts.items():
            print(f"  {c:16s} {n:4d}  ({100*n/total:.1f}%)")


if __name__ == "__main__":
    main()
