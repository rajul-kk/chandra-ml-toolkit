"""Domain-agnostic pool-based active learning.

Works against any scikit-learn-style estimator (fit / predict / predict_proba)
and a plain numpy feature pool - nothing here knows about Chandra, X-ray
features, or class names. catalog_classification supplies the estimator,
the feature matrix, and (for the reliability-aware experiment) an auxiliary
per-example array; this module just runs the query loop.

Query strategies are plain scoring functions, higher score = more worth
labeling:

    score_fn(estimator, X_labeled, y_labeled, X_pool, pool_indices, rng) -> np.ndarray

`pool_indices` are the caller's original dataset indices for the rows
currently in X_pool (the pool shrinks as rounds proceed, so this is how a
strategy looks up per-example side information, e.g. reliability, keyed to
the original dataset rather than the current pool position).

`reliability_weighted()` wraps any base strategy with a per-example
reliability array (any domain's, not just Chandra's) and a mixing exponent
alpha - this is the method-layer contribution, not a Chandra-specific hack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.utils import check_random_state

ScoreFn = Callable[..., np.ndarray]


# ---------------------------------------------------------------------------
# Query strategies
# ---------------------------------------------------------------------------

def uncertainty_score(estimator, X_labeled, y_labeled, X_pool, pool_indices=None, rng=None, **kw):
    """Least-confidence: 1 - P(predicted class)."""
    proba = estimator.predict_proba(X_pool)
    return 1.0 - proba.max(axis=1)


def margin_score(estimator, X_labeled, y_labeled, X_pool, pool_indices=None, rng=None, **kw):
    """Smallest margin between the top two predicted-class probabilities."""
    proba = np.sort(estimator.predict_proba(X_pool), axis=1)
    return -(proba[:, -1] - proba[:, -2])


def qbc_score(estimator, X_labeled, y_labeled, X_pool, pool_indices=None, rng=None,
              n_committee: int = 5, sample_frac: float = 0.8, **kw):
    """Query-by-committee: vote entropy across bagged committee members."""
    rng = check_random_state(rng)
    n_labeled = len(X_labeled)
    if n_labeled < 2:
        return rng.random(X_pool.shape[0])
    votes = np.empty((n_committee, X_pool.shape[0]), dtype=object)
    for m in range(n_committee):
        idx = rng.choice(n_labeled, size=max(2, int(n_labeled * sample_frac)), replace=True)
        member = clone(estimator)
        member.fit(X_labeled[idx], y_labeled[idx])
        votes[m] = member.predict(X_pool)
    scores = np.zeros(X_pool.shape[0])
    classes = np.unique(votes)
    for i in range(X_pool.shape[0]):
        counts = np.array([(votes[:, i] == c).sum() for c in classes])
        p = counts[counts > 0] / n_committee
        scores[i] = -(p * np.log(p)).sum()
    return scores


def class_balanced_uncertainty_score(estimator, X_labeled, y_labeled, X_pool, pool_indices=None,
                                      rng=None, **kw):
    """Uncertainty sampling, but ranked within each predicted class rather
    than globally.

    Plain uncertainty_score picks the top-N most-uncertain pool examples
    across all classes at once; when one class dominates the pool's raw
    count, its boundary-adjacent examples dominate that top-N by sheer
    numbers, and a numerically tiny class can go entirely unrepresented in
    every query batch even though it may be the class most worth labeling.
    Converting each example's uncertainty to a percentile *within its own
    predicted class* puts every class on the same [0, 1] scale regardless
    of how many pool examples currently fall into it, so a batch selected
    by top-N on this score draws roughly evenly across predicted classes
    instead of being crowded out by the largest one.
    """
    proba = estimator.predict_proba(X_pool)
    raw_uncertainty = 1.0 - proba.max(axis=1)
    pred_class = proba.argmax(axis=1)
    scores = np.zeros_like(raw_uncertainty)
    for c in np.unique(pred_class):
        mask = pred_class == c
        n_c = mask.sum()
        if n_c <= 1:
            scores[mask] = 1.0
            continue
        ranks = raw_uncertainty[mask].argsort().argsort()
        scores[mask] = ranks / (n_c - 1)
    return scores


def quota_score(estimator, X_labeled, y_labeled, X_pool, pool_indices=None, rng=None,
                 batch_size=20, base_score_fn=None, min_frac_per_class=None, **kw):
    """Hard-quota acquisition: reserve a guaranteed minimum share of every
    query batch for each class, filled by that class's own top-probability
    candidates, with any remaining slots filled by `base_score_fn` globally.

    class_balanced_uncertainty_score's percentile-within-predicted-class
    ranking is a *soft* nudge - it only guarantees the single top-ranked
    member of a class survives top-N selection, which in practice barely
    moved a severely rare class's representation (checked empirically: 33
    vs 39 queries out of 800 for a ~3%-of-pool class). This strategy makes
    the guarantee explicit and hard instead of relying on percentile scores
    surviving a global top-N cut: `min_frac_per_class` slots per class are
    reserved outright, ranked within that reservation by predicted P(class)
    rather than by predicted-class-bucket membership (avoiding
    class_balanced's failure mode of an argmax bucket contaminated by
    low-confidence noise).
    """
    base_score_fn = base_score_fn or uncertainty_score
    base_scores = base_score_fn(estimator, X_labeled, y_labeled, X_pool,
                                 pool_indices=pool_indices, rng=rng, **kw)
    proba = estimator.predict_proba(X_pool)
    n_classes = proba.shape[1]
    if min_frac_per_class is None:
        min_frac_per_class = 1.0 / n_classes

    n_reserved_per_class = min(int(round(min_frac_per_class * batch_size)), batch_size // n_classes)
    scores = base_scores.copy()
    if n_reserved_per_class > 0:
        # reserved picks must outrank every non-reserved example under top-N
        # selection; tie-break within the reservation by base_score so the
        # method still prefers the more informative members of each class
        bonus = (np.nanmax(base_scores) - np.nanmin(base_scores) if len(base_scores) else 0) + 1.0
        for c in range(n_classes):
            top_c = np.argsort(-proba[:, c])[:n_reserved_per_class]
            scores[top_c] = bonus + base_scores[top_c]
    return scores


def prototype_distance_score(estimator, X_labeled, y_labeled, X_pool, pool_indices=None, rng=None,
                              batch_size=20, quota_frac=None, k_neighbors=5,
                              base_score_fn=None, **kw):
    """Classifier-independent acquisition for the rarest currently-labeled
    class: score pool examples by feature-space proximity to that class's
    own labeled members, entirely bypassing the classifier's predict_proba.

    Every other strategy in this module - uncertainty, margin,
    class_balanced, quota, reliability_weighted - ultimately ranks pool
    examples using this classifier's own probability estimates. Checked
    empirically (see chandra-toolkit's catalog_classification results):
    under severe class imbalance with few labels, those estimates are
    barely discriminative for the rare class, and every probability-based
    strategy inherits that blindness, including a hard quota reserving
    slots for it explicitly. This strategy tries a different signal:
    raw distance in (z-scored) feature space to already-labeled examples
    of whichever class currently has the fewest labels, on the premise
    that "looks like the other members of this rare class" doesn't
    require the classifier to have learned anything about that class yet.

    Reserves `quota_frac` of the batch (default 1/n_classes) for the
    pool examples closest to that class's labeled prototypes (mean
    distance to their `k_neighbors` nearest labeled members), with the
    rest of the batch filled by `base_score_fn` (default uncertainty_score)
    exactly as in quota_score. Falls back to `base_score_fn` alone if the
    rarest class has no labeled examples yet to build prototypes from.
    """
    base_score_fn = base_score_fn or uncertainty_score
    base_scores = base_score_fn(estimator, X_labeled, y_labeled, X_pool,
                                 pool_indices=pool_indices, rng=rng, **kw)

    classes, counts = np.unique(y_labeled, return_counts=True)
    if len(classes) == 0:
        return base_scores
    rarest_class = classes[np.argmin(counts)]
    X_rare = X_labeled[y_labeled == rarest_class]
    if len(X_rare) == 0:
        return base_scores

    # z-score using combined labeled+pool statistics so distance isn't
    # dominated by whichever raw feature happens to have the largest scale
    combined = np.vstack([X_labeled, X_pool])
    mu = combined.mean(axis=0)
    sigma = combined.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_rare_z = (X_rare - mu) / sigma
    X_pool_z = (X_pool - mu) / sigma

    k = min(k_neighbors, len(X_rare_z))
    dists = cdist(X_pool_z, X_rare_z)
    nearest_k = np.sort(dists, axis=1)[:, :k]
    proximity = -nearest_k.mean(axis=1)  # higher = closer to the rare class's labeled members

    n_classes = len(classes)
    quota_frac = quota_frac if quota_frac is not None else 1.0 / n_classes
    n_reserved = min(int(round(quota_frac * batch_size)), batch_size)

    scores = base_scores.copy()
    if n_reserved > 0:
        top_prototype = np.argsort(-proximity)[:n_reserved]
        bonus = (np.nanmax(base_scores) - np.nanmin(base_scores) if len(base_scores) else 0) + 1.0
        scores[top_prototype] = bonus + proximity[top_prototype]
    return scores


def random_score(estimator, X_labeled, y_labeled, X_pool, pool_indices=None, rng=None, **kw):
    """Control strategy: every example equally worth labeling."""
    rng = check_random_state(rng)
    return rng.random(X_pool.shape[0])


def reliability_weighted(base_score_fn: ScoreFn, reliability: np.ndarray, alpha: float = 1.0) -> ScoreFn:
    """Wrap a strategy so its score is discounted for examples whose label
    is expected to be less trustworthy.

    `reliability` is indexed by the caller's original dataset indices (same
    space as `pool_indices`), values expected roughly in [0, 1]. alpha=0
    recovers the base strategy exactly; alpha=1 multiplies informativeness
    by reliability; alpha>1 penalizes low-reliability examples harder.
    """
    def score_fn(estimator, X_labeled, y_labeled, X_pool, pool_indices=None, rng=None, **kw):
        raw = base_score_fn(estimator, X_labeled, y_labeled, X_pool,
                             pool_indices=pool_indices, rng=rng, **kw)
        if pool_indices is None:
            raise ValueError("reliability_weighted requires pool_indices to look up reliability")
        rel = np.asarray(reliability)[pool_indices]
        return raw * (rel ** alpha)
    return score_fn


# ---------------------------------------------------------------------------
# Active learning loop
# ---------------------------------------------------------------------------

@dataclass
class LearningHistory:
    strategy_name: str
    n_labels: List[int] = field(default_factory=list)
    metrics: List[dict] = field(default_factory=list)
    queried_indices: List[np.ndarray] = field(default_factory=list)


class ActiveLearner:
    """Pool-based active learning loop.

    Parameters
    ----------
    estimator : sklearn-style estimator (must support fit, predict_proba)
    X : full feature matrix (labeled seed rows + unlabeled pool rows)
    label_fn : callable(indices) -> labels ; the oracle. In this project it
        reads a held-out ground-truth array, but the signature is what a
        human-in-the-loop labeling callback would look like too.
    score_fn : one of the strategies above, or a custom callable
    init_indices : indices into X that start out labeled
    eval_fn : optional callable(fitted_estimator) -> dict of metrics,
        called once per round for the label-efficiency curve
    """

    def __init__(self, estimator, X: np.ndarray, label_fn: Callable[[np.ndarray], np.ndarray],
                 score_fn: ScoreFn, init_indices: np.ndarray, batch_size: int = 20,
                 eval_fn: Optional[Callable[[object], dict]] = None,
                 strategy_name: str = "custom", random_state=None):
        self.estimator = estimator
        self.X = np.asarray(X)
        self.label_fn = label_fn
        self.score_fn = score_fn
        self.batch_size = batch_size
        self.eval_fn = eval_fn
        self.strategy_name = strategy_name
        self.rng = check_random_state(random_state)

        self.labeled_idx = np.array(sorted(set(init_indices)), dtype=int)
        all_idx = np.arange(len(self.X))
        self.pool_idx = np.setdiff1d(all_idx, self.labeled_idx)
        self.y_labeled = np.asarray(label_fn(self.labeled_idx))

    def _fit_current(self):
        model = clone(self.estimator)
        model.fit(self.X[self.labeled_idx], self.y_labeled)
        return model

    def step(self, model) -> np.ndarray:
        """Score the current pool, pick a batch, label it, fold it into the
        labeled set. Returns the original-index array that was queried.
        """
        if len(self.pool_idx) == 0:
            return np.array([], dtype=int)
        X_pool = self.X[self.pool_idx]
        n = min(self.batch_size, len(self.pool_idx))
        scores = self.score_fn(model, self.X[self.labeled_idx], self.y_labeled, X_pool,
                                pool_indices=self.pool_idx, rng=self.rng, batch_size=n)
        pick_pos = np.argpartition(-scores, n - 1)[:n]
        queried = self.pool_idx[pick_pos]

        new_labels = np.asarray(self.label_fn(queried))
        self.labeled_idx = np.concatenate([self.labeled_idx, queried])
        self.y_labeled = np.concatenate([self.y_labeled, new_labels])
        self.pool_idx = np.setdiff1d(self.pool_idx, queried)
        return queried

    def run(self, n_rounds: int) -> LearningHistory:
        history = LearningHistory(strategy_name=self.strategy_name)
        model = self._fit_current()
        if self.eval_fn is not None:
            history.n_labels.append(len(self.labeled_idx))
            history.metrics.append(self.eval_fn(model))
            history.queried_indices.append(self.labeled_idx.copy())

        for _ in range(n_rounds):
            queried = self.step(model)
            if len(queried) == 0:
                break
            model = self._fit_current()
            if self.eval_fn is not None:
                history.n_labels.append(len(self.labeled_idx))
                history.metrics.append(self.eval_fn(model))
                history.queried_indices.append(queried.copy())
        return history
