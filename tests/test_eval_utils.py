from pathlib import Path

import pytest

from common.active_learning import LearningHistory
from common.eval_utils import (
    average_histories,
    label_savings_fraction,
    labels_to_match,
    write_csv,
)


def make_history(name, n_labels, accuracies):
    return LearningHistory(
        strategy_name=name,
        n_labels=n_labels,
        metrics=[{"accuracy": a} for a in accuracies],
        queried_indices=[],
    )


def test_write_csv_produces_one_row_per_round(tmp_path):
    h1 = make_history("al", [10, 20, 30], [0.5, 0.6, 0.7])
    h2 = make_history("random", [10, 20, 30], [0.4, 0.5, 0.55])
    out = tmp_path / "log.csv"
    write_csv([h1, h2], out)
    text = out.read_text()
    lines = text.strip().splitlines()
    assert len(lines) == 1 + 6  # header + 3 rounds * 2 strategies
    assert "strategy" in lines[0] and "n_labels" in lines[0] and "accuracy" in lines[0]


def test_average_histories_averages_across_seeds():
    h1 = make_history("al", [10, 20], [0.5, 0.7])
    h2 = make_history("al", [10, 20], [0.6, 0.9])
    avg = average_histories([h1, h2])
    assert avg["n_labels"] == [10, 20]
    assert avg["accuracy"]["mean"] == pytest.approx([0.55, 0.8])


def test_average_histories_rejects_mismatched_schedules():
    h1 = make_history("al", [10, 20], [0.5, 0.7])
    h2 = make_history("al", [10, 30], [0.6, 0.9])
    with pytest.raises(ValueError):
        average_histories([h1, h2])


def test_labels_to_match_finds_first_crossing():
    candidate = {"n_labels": [10, 20, 30], "accuracy": {"mean": [0.5, 0.8, 0.9], "std": [0, 0, 0]}}
    reference = {"n_labels": [50], "accuracy": {"mean": [0.79], "std": [0]}}
    assert labels_to_match(candidate, reference, "accuracy") == 20


def test_labels_to_match_returns_none_if_never_reached():
    candidate = {"n_labels": [10, 20], "accuracy": {"mean": [0.1, 0.2], "std": [0, 0]}}
    reference = {"n_labels": [50], "accuracy": {"mean": [0.9], "std": [0]}}
    assert labels_to_match(candidate, reference, "accuracy") is None


def test_label_savings_fraction_computes_expected_saving():
    # candidate matches reference's final score at 20 labels; reference used 50
    candidate = {"n_labels": [10, 20, 30], "accuracy": {"mean": [0.5, 0.8, 0.9], "std": [0, 0, 0]}}
    reference = {"n_labels": [50], "accuracy": {"mean": [0.79], "std": [0]}}
    saving = label_savings_fraction(candidate, reference, "accuracy")
    assert saving == pytest.approx(1 - 20 / 50)
