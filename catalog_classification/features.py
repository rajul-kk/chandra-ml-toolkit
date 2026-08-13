"""Turn the joined labeled pool (catalog_classification.dataset) into a
numeric feature matrix for classification.

Missingness in hard_hm/hard_hs/hard_ms and var_inter_index_b is physical,
not random (faint sources lack full-band detections; most sources lack a
multi-epoch variability measurement) - each gets a missing-indicator
column rather than being silently imputed away, since "no signal in this
band" and "measured as zero" are different facts a classifier should be
able to use.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

LABELS = ["AGN", "STAR", "COMPACT_OBJECT"]

NUMERIC_COLS = [
    "significance", "flux_aper_b", "hard_hm", "hard_hs", "hard_ms",
    "var_intra_index_b", "var_inter_index_b", "separation",
    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
]
BOOL_COLS = ["extent_flag", "conf_flag", "flag_nway_confident", "flag_ml_confident"]
RELIABILITY_COL = "p_match_ind"


def _add_gaia_colors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"phot_bp_mean_mag", "phot_rp_mean_mag"}.issubset(df.columns):
        df["bp_rp"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]
    if {"phot_g_mean_mag", "phot_rp_mean_mag"}.issubset(df.columns):
        df["g_rp"] = df["phot_g_mean_mag"] - df["phot_rp_mean_mag"]
    return df


def build_feature_matrix(pool: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Returns (X, y, feature_names, reliability).

    X: numeric feature matrix, missing-value flags added for
       physically-missing columns, NaNs filled with column median.
    y: integer-encoded labels (index into LABELS).
    reliability: p_match_ind per row, for the reliability-aware strategy.
    """
    df = _add_gaia_colors(pool)
    numeric_cols = [c for c in NUMERIC_COLS + ["bp_rp", "g_rp"] if c in df.columns]

    feature_names: List[str] = []
    columns = []

    for col in numeric_cols:
        values = df[col].to_numpy(dtype=float)
        missing = np.isnan(values)
        if missing.any():
            median = np.nanmedian(values)
            values = np.where(missing, median, values)
            columns.append(missing.astype(float))
            feature_names.append(f"{col}__missing")
        columns.append(values)
        feature_names.append(col)

    for col in BOOL_COLS:
        if col in df.columns:
            columns.append(df[col].astype(float).to_numpy())
            feature_names.append(col)

    X = np.column_stack(columns)
    y = df["label"].map(LABELS.index).to_numpy()
    reliability = df[RELIABILITY_COL].to_numpy(dtype=float)
    return X, y, feature_names, reliability
