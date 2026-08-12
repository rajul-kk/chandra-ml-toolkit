"""Tests for catalog_classification.dataset, run against tiny local fixture
CSVs (not the full downloaded catalogs) so they run fast and offline.
"""
from pathlib import Path

import pandas as pd
import pytest

from catalog_classification import dataset

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def td_path(tmp_path):
    df = pd.DataFrame({
        "name": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "Class": ["AGN", "YSO", "LM-STAR", "NS", "CV", "QSO"],  # QSO unmapped
    })
    p = tmp_path / "td.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def matches_path(tmp_path):
    df = pd.DataFrame({
        "csc21_name": ["S1", "S2", "S3", "S4", "S7"],  # S7 has no label
        "p_i": [1.0, 0.5, 1.0, 0.2, 0.9],
        "p_any": [0.99, 0.6, 0.95, 0.3, 0.9],
        "p_match_ind": [0.9, 0.5, 0.8, 0.47, 0.9],
        "separation": [0.1, 1.2, 0.3, 2.0, 0.2],
        "flag_nway_confident": [True, False, True, False, True],
        "flag_ml_confident": [True, True, True, True, True],
    })
    p = tmp_path / "matches.csv"
    df.to_csv(p, index=False)
    return p


def test_load_labels_maps_known_classes_and_drops_unmapped(td_path):
    labels = dataset.load_labels(td_path)
    assert set(labels["name"]) == {"S1", "S2", "S3", "S4", "S5"}
    assert labels.set_index("name").loc["S1", "label"] == "AGN"
    assert labels.set_index("name").loc["S2", "label"] == "STAR"
    assert labels.set_index("name").loc["S4", "label"] == "COMPACT_OBJECT"


def test_load_labeled_pool_inner_joins_and_keeps_reliability_cols(td_path, matches_path):
    pool = dataset.load_labeled_pool(td_path, matches_path)
    # S5 has no match row, S6 unmapped class, S7 has no label -> only S1-S4 survive
    assert set(pool["name"]) == {"S1", "S2", "S3", "S4"}
    assert "p_any" in pool.columns
    assert "label" in pool.columns
    assert pool.set_index("name").loc["S4", "label"] == "COMPACT_OBJECT"
