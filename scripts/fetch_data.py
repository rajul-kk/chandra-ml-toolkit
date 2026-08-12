"""Download the raw catalog files this project's experiments are built on.

Sources (both public, both re-downloadable, neither committed to git):

1. Chandra-Gaia Catalog of Counterparts (Perez-Diaz et al. 2026,
   arXiv:2606.19329) - per-source counterpart-match reliability scores
   (p_i, p_any, p_match_ind) for ~113k CSC 2.1 X-ray sources.
   Zenodo DOI 10.5281/zenodo.18652667.

2. MUWCLASS training dataset (Yang et al. 2022, arXiv:2206.13656) -
   ~2,962 CSC 2.1 sources with SIMBAD-derived class labels (AGN, YSO,
   LM-STAR, HM-STAR, NS, CV, LMXB, HMXB, NS_BIN).
   https://github.com/huiyang-astro/MUWCLASS_CSCv2

Run: python scripts/fetch_data.py
"""
from pathlib import Path
import urllib.request

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FILES = {
    "csc_gaia_best_matches.csv":
        "https://zenodo.org/api/records/18652667/files/csc_gaia_best_matches.csv/content",
    "csc_gaia_ambiguous_nway_matches.csv":
        "https://zenodo.org/api/records/18652667/files/csc_gaia_ambiguous_nway_matches.csv/content",
    "csc_gaia_alternative_ml_matches.csv":
        "https://zenodo.org/api/records/18652667/files/csc_gaia_alternative_ml_matches.csv/content",
    "muwclass_td.csv":
        "https://raw.githubusercontent.com/huiyang-astro/MUWCLASS_CSCv2/main/files/CSC_TD_MW_remove.csv",
}


def fetch_all(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW_DIR / name
        if dest.exists() and not force:
            print(f"skip (exists): {name}")
            continue
        print(f"downloading: {name}")
        urllib.request.urlretrieve(url, dest)
    print("done.")


if __name__ == "__main__":
    fetch_all()
