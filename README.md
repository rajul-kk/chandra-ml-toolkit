# chandra-ml-toolkit

Three projects attacking the Chandra X-ray archive through its three native
data products, sharing one data-access and active-learning foundation:

| Module | Data product | Status |
|---|---|---|
| [`catalog_classification/`](catalog_classification/) | catalog features (CSC 2.1) | active |
| `image_anomaly_detection/` | image cutouts (AnomalyMatch-based) | planned |
| `eventfile_representation/` | raw event files (representation learning) | planned |

The shared modules in [`common/`](common/) are built generally from day one
so the later two projects extend rather than duplicate this one's plumbing:

- **`common/data_access.py`** - resolves a CSC source to its catalog row,
  image cutout, and event file. Only the catalog row is implemented end to
  end (that's all this project needs); cutout and event-file access are
  stubbed with the query each later module should issue, against the same
  TAP service (`csc21.image`, `ivoa.ObsCore`) this project already queries.
  Includes a disk cache keyed by source id / query hash, since all three
  projects will re-query overlapping sets of sources.
- **`common/active_learning.py`** - a pool-based active-learning loop,
  estimator-agnostic (anything with `fit`/`predict_proba`) and
  domain-agnostic (plain numpy in, no Chandra-specific feature names).
  Strategies: uncertainty, margin, query-by-committee, random (control),
  and a `reliability_weighted` wrapper that discounts acquisition by any
  per-example reliability signal the caller supplies.
- **`common/eval_utils.py`** - label-efficiency curve plotting, CSV
  logging, and the labels-to-match / label-savings calculation, reusable
  for any future label-efficiency comparison in this repo.

## Project 1: active-learning label efficiency for X-ray source classification

**Question:** how many labels does active learning save over random
sampling on CSC 2.1 source-type classification, the way RB-C1000
(Liu et al. 2025, arXiv:2412.02409) showed for ZTF real/bogus
classification (~60% savings)?

**Data:** CSC 2.1 X-ray features (hardness ratios, flux, variability,
significance) via TAP, joined to Gaia DR3 optical photometry and
counterpart-match reliability scores from the Chandra-Gaia Catalog of
Counterparts (Perez-Diaz et al. 2026, arXiv:2606.19329), labeled with
SIMBAD-derived classes from the MUWCLASS training set (Yang et al. 2022,
arXiv:2206.13656). ~2,191 sources with both a class label and a resolved
counterpart, 3-class (AGN / STAR / COMPACT_OBJECT - no "galaxy" class
exists in X-ray point-source catalogs).

**Novelty (checked August 2026):** existing Chandra/eROSITA
counterpart-classification pipelines (the Chandra-Gaia catalog above,
the eRASS1 counterpart/AGN paper arXiv:2509.02842) all train on fixed,
pre-curated label sets. None apply active learning to reduce the label
budget.

**Methods contribution:** the same acquisition function that makes
uncertainty sampling effective - querying sources nearest the decision
boundary - also concentrates queries on the sources whose counterpart
match is itself ambiguous, since positional/photometric ambiguity drives
both classifier uncertainty and match unreliability. Standard AL assumes
a clean oracle; here the label noise is correlated with what AL chooses
to query, which is close to worst-case for active learning
(see arXiv:2607.13233 on label noise in X-ray AGN/SFG classification).
The Chandra-Gaia catalog's `p_match_ind` gives a usable per-source
reliability signal for free, letting the acquisition function be
corrected for it directly.

**Results:** see `results/` for the label-efficiency curve and
`catalog_classification/analyze_reliability_correlation.py` for the
direct measurement of the noise-acquisition correlation.
_(filled in once the full run completes)_

## Setup

```
pip install -r requirements.txt
python scripts/fetch_data.py
python -m pytest
python -m catalog_classification.run_experiment
```
