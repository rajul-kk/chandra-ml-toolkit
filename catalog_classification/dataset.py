"""Build the labeled pool for this project: CSC 2.1 sources with a class
label (from MUWCLASS) and a counterpart-match reliability score (from the
Chandra-Gaia Catalog of Counterparts).

The label scheme is 3-class, not the originally-planned 4-class
star/AGN/galaxy/compact-object split: MUWCLASS carries no "galaxy" label
because resolved galaxies are not X-ray point sources in this catalog.
Real classes collapse to:

    AGN              <- AGN
    STAR             <- LM-STAR, HM-STAR, YSO
    COMPACT_OBJECT   <- NS, NS_BIN, CV, LMXB, HMXB

See data/README.md for provenance and scripts/fetch_data.py to populate
data/raw/.
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

CLASS_MAP = {
    "AGN": "AGN",
    "LM-STAR": "STAR",
    "HM-STAR": "STAR",
    "YSO": "STAR",
    "NS": "COMPACT_OBJECT",
    "NS_BIN": "COMPACT_OBJECT",
    "CV": "COMPACT_OBJECT",
    "LMXB": "COMPACT_OBJECT",
    "HMXB": "COMPACT_OBJECT",
}


def load_labels(td_path: Path = None) -> pd.DataFrame:
    """MUWCLASS training-set class labels, keyed by CSC source name."""
    td_path = td_path or RAW_DIR / "muwclass_td.csv"
    td = pd.read_csv(td_path, usecols=["name", "Class"], low_memory=False)
    td = td[td["Class"].isin(CLASS_MAP)].copy()
    td["label"] = td["Class"].map(CLASS_MAP)
    return td[["name", "label"]].drop_duplicates(subset="name")


def load_reliability(matches_path: Path = None) -> pd.DataFrame:
    """Per-source counterpart-match reliability scores, keyed by CSC name."""
    matches_path = matches_path or RAW_DIR / "csc_gaia_best_matches.csv"
    cols = ["csc21_name", "p_i", "p_any", "p_match_ind", "separation",
            "flag_nway_confident", "flag_ml_confident"]
    m = pd.read_csv(matches_path, usecols=cols)
    return m.rename(columns={"csc21_name": "name"})


def load_labeled_pool(td_path: Path = None, matches_path: Path = None) -> pd.DataFrame:
    """Join labels to reliability scores. Inner join: a source only enters
    the pool for this project if it has both a known class and a resolved
    counterpart match (the pool the label-efficiency study samples from).
    """
    labels = load_labels(td_path)
    reliability = load_reliability(matches_path)
    pool = labels.merge(reliability, on="name", how="inner")
    return pool.reset_index(drop=True)
