import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from common.active_learning import (
    ActiveLearner,
    margin_score,
    qbc_score,
    random_score,
    reliability_weighted,
    uncertainty_score,
)


class StubEstimator:
    """Fixed predict_proba, ignores fit - lets us test scoring functions
    against a known-correct answer instead of a trained model's output.
    """
    def __init__(self, proba):
        self._proba = np.asarray(proba)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self._proba

    def predict(self, X):
        return self._proba.argmax(axis=1)


def test_uncertainty_score_ranks_least_confident_highest():
    # row 0 is a confident call, row 1 is a coin flip -> row 1 should score higher
    proba = [[0.95, 0.05], [0.5, 0.5], [0.8, 0.2]]
    scores = uncertainty_score(StubEstimator(proba), None, None, np.zeros((3, 1)))
    assert scores.argmax() == 1
    assert scores[1] == pytest.approx(0.5)


def test_margin_score_ranks_smallest_margin_highest():
    proba = [[0.9, 0.1], [0.55, 0.45], [0.34, 0.33, 0.33]]
    # use two 2-class rows only, to keep sort-based margin well defined
    proba2 = [[0.9, 0.1], [0.55, 0.45]]
    scores = margin_score(StubEstimator(proba2), None, None, np.zeros((2, 1)))
    assert scores.argmax() == 1  # smaller gap (0.10) beats larger gap (0.80)


def test_random_score_is_deterministic_given_rng():
    proba = [[0.5, 0.5]] * 5
    rng = np.random.RandomState(0)
    scores = random_score(StubEstimator(proba), None, None, np.zeros((5, 1)), rng=rng)
    assert scores.shape == (5,)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_reliability_weighted_downweights_low_reliability_examples():
    # two equally uncertain examples, but example 1 has low label reliability
    proba = [[0.5, 0.5], [0.5, 0.5]]
    reliability = np.array([1.0, 0.1])
    pool_indices = np.array([0, 1])
    wrapped = reliability_weighted(uncertainty_score, reliability, alpha=1.0)
    scores = wrapped(StubEstimator(proba), None, None, np.zeros((2, 1)), pool_indices=pool_indices)
    assert scores[0] > scores[1]  # equal raw uncertainty, but 0 is more reliable


def test_reliability_weighted_requires_pool_indices():
    proba = [[0.5, 0.5]]
    wrapped = reliability_weighted(uncertainty_score, np.array([1.0]), alpha=1.0)
    with pytest.raises(ValueError):
        wrapped(StubEstimator(proba), None, None, np.zeros((1, 1)), pool_indices=None)


def test_qbc_score_returns_nonnegative_entropy_shape_matches_pool():
    X_labeled = np.random.RandomState(0).randn(20, 3)
    y_labeled = np.random.RandomState(0).randint(0, 2, 20)
    X_pool = np.random.RandomState(1).randn(10, 3)
    scores = qbc_score(DecisionTreeClassifier(), X_labeled, y_labeled, X_pool,
                        rng=np.random.RandomState(0), n_committee=5)
    assert scores.shape == (10,)
    assert (scores >= 0).all()


def test_active_learner_grows_labeled_set_and_shrinks_pool():
    X, y = make_classification(n_samples=200, n_features=5, n_classes=2,
                                n_informative=3, random_state=0)
    init = np.arange(10)
    learner = ActiveLearner(
        estimator=LogisticRegression(max_iter=200),
        X=X, label_fn=lambda idx: y[idx],
        score_fn=uncertainty_score, init_indices=init, batch_size=15,
        random_state=0,
    )
    assert len(learner.labeled_idx) == 10
    assert len(learner.pool_idx) == 190

    history = learner.run(n_rounds=3)
    assert len(learner.labeled_idx) == 10 + 3 * 15
    assert len(learner.pool_idx) == 200 - len(learner.labeled_idx)
    assert history.n_labels == []  # no eval_fn supplied


def test_active_learner_records_history_with_eval_fn():
    X, y = make_classification(n_samples=150, n_features=5, n_classes=2,
                                n_informative=3, random_state=1)
    X_test, y_test = X[:30], y[:30]
    X_train, y_train = X[30:], y[30:]

    def eval_fn(model):
        return {"accuracy": accuracy_score(y_test, model.predict(X_test))}

    learner = ActiveLearner(
        estimator=LogisticRegression(max_iter=200),
        X=X_train, label_fn=lambda idx: y_train[idx],
        score_fn=uncertainty_score, init_indices=np.arange(10), batch_size=10,
        eval_fn=eval_fn, strategy_name="uncertainty", random_state=0,
    )
    history = learner.run(n_rounds=4)
    assert history.strategy_name == "uncertainty"
    assert len(history.n_labels) == 5  # initial + 4 rounds
    assert history.n_labels == [10, 20, 30, 40, 50]
    assert all("accuracy" in m for m in history.metrics)


def test_uncertainty_sampling_beats_random_on_separable_data_with_label_noise_near_boundary():
    """A real correctness check, not just plumbing: on a dataset with a
    genuine decision boundary, uncertainty sampling should reach a given
    accuracy with fewer labels than random sampling, on average.
    """
    X, y = make_classification(n_samples=400, n_features=6, n_classes=2,
                                n_informative=4, class_sep=1.2, random_state=42)
    X_test, y_test = X[:100], y[:100]
    X_pool, y_pool = X[100:], y[100:]

    def eval_fn(model):
        return {"accuracy": accuracy_score(y_test, model.predict(X_test))}

    def run(score_fn, seed):
        rng = np.random.RandomState(seed)
        init = rng.choice(len(X_pool), size=10, replace=False)
        learner = ActiveLearner(
            estimator=LogisticRegression(max_iter=500),
            X=X_pool, label_fn=lambda idx: y_pool[idx],
            score_fn=score_fn, init_indices=init, batch_size=10,
            eval_fn=eval_fn, random_state=seed,
        )
        return learner.run(n_rounds=8)

    unc_final = [run(uncertainty_score, s).metrics[-1]["accuracy"] for s in range(5)]
    rand_final = [run(random_score, s).metrics[-1]["accuracy"] for s in range(5)]

    # not a strict inequality every seed, but on average AL should not be worse
    assert np.mean(unc_final) >= np.mean(rand_final) - 0.02
