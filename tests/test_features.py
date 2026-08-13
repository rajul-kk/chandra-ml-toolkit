import numpy as np
import pandas as pd

from catalog_classification.features import LABELS, build_feature_matrix


def make_pool():
    return pd.DataFrame({
        "name": ["S1", "S2", "S3"],
        "label": ["AGN", "STAR", "COMPACT_OBJECT"],
        "p_match_ind": [0.9, 0.5, 0.3],
        "significance": [10.0, 5.0, np.nan],
        "flux_aper_b": [1e-14, 2e-14, 3e-14],
        "hard_hm": [0.1, np.nan, -0.2],
        "hard_hs": [0.1, 0.2, -0.1],
        "hard_ms": [0.1, 0.2, -0.1],
        "var_intra_index_b": [0, 1, 2],
        "var_inter_index_b": [np.nan, np.nan, 3.0],
        "separation": [0.1, 0.2, 0.3],
        "phot_g_mean_mag": [15.0, 16.0, 17.0],
        "phot_bp_mean_mag": [15.5, 16.5, 17.5],
        "phot_rp_mean_mag": [14.5, 15.5, 16.5],
        "extent_flag": [False, True, False],
        "conf_flag": [False, False, True],
        "flag_nway_confident": [True, True, False],
        "flag_ml_confident": [True, True, True],
    })


def test_build_feature_matrix_shapes_and_no_nans():
    X, y, names, reliability = build_feature_matrix(make_pool())
    assert X.shape[0] == 3
    assert not np.isnan(X).any()
    assert len(names) == X.shape[1]
    assert list(y) == [LABELS.index("AGN"), LABELS.index("STAR"), LABELS.index("COMPACT_OBJECT")]
    assert list(reliability) == [0.9, 0.5, 0.3]


def test_missing_columns_get_indicator_and_median_fill():
    X, y, names, _ = build_feature_matrix(make_pool())
    assert "significance__missing" in names
    assert "hard_hm__missing" in names
    assert "var_inter_index_b__missing" in names

    sig_idx = names.index("significance")
    miss_idx = names.index("significance__missing")
    # row 2 had NaN significance -> filled with median of [10, 5] = 7.5, flag=1
    assert X[2, sig_idx] == 7.5
    assert X[2, miss_idx] == 1.0
    assert X[0, miss_idx] == 0.0


def test_gaia_colors_are_computed():
    X, y, names, _ = build_feature_matrix(make_pool())
    assert "bp_rp" in names
    assert "g_rp" in names
    bp_rp_idx = names.index("bp_rp")
    assert X[0, bp_rp_idx] == 1.0  # 15.5 - 14.5
