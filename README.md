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

**Results (negative/mixed finding, 15 seeds, LightGBM, macro-F1):**

![label efficiency curve](results/label_efficiency_curve.png)

Active learning does **not** produce a statistically significant label
saving over random sampling on this pool. All four strategies (random,
uncertainty, margin, reliability-weighted) converge to the same macro-F1
plateau (~0.637-0.643) by roughly 150-300 labels and stay statistically
indistinguishable out to the full 830-label budget (paired comparison at
the plateau, n=75 pooled samples per strategy: p=0.19 uncertainty,
p=0.26 margin, p=0.44 reliability-weighted vs. random - none below 0.05).

This is well below the ~15-20% saving the original scoping treated as the
threshold for a positive result, let alone RB-C1000's ~60%. An early,
naive read of this experiment (5 seeds, comparing each strategy to
random's literal final round) reported 80%+ "savings" - that was an
artifact of random sampling's macro-F1 plateauing and then oscillating
rather than climbing, which makes "labels needed to match the final
round" measure noise, not real efficiency, once a curve has flattened.
Fixing `eval_utils.labels_to_match` to target a tail-window average
(see `common/eval_utils.py`) and re-running at 15 seeds for power
resolved it into the null result above.

**Layer 2 (why):** the mechanism motivating reliability-weighted
acquisition - that classifier uncertainty and counterpart-match
unreliability are correlated, so vanilla uncertainty sampling
preferentially queries untrustworthy labels - does hold at the
*class* level (COMPACT_OBJECT's mean `p_match_ind` is 0.82 vs. AGN's
0.95) but is essentially absent at the *acquisition* level: the
point-biserial correlation between "queried by uncertainty sampling"
and match reliability is r=-0.03 (p=0.19,
`catalog_classification/analyze_reliability_correlation.py`), i.e. not
distinguishable from zero. Uncertainty sampling does mildly
overrepresent the rare COMPACT_OBJECT class in what it queries (4.9%
of queries vs. 3.4% of the pool) but not by preferentially picking its
least-reliable members - which is consistent with reliability-weighted
acquisition showing no benefit: there's little acquisition-level noise
correlation for it to correct.

**Honest interpretation:** on this pool (2,191 sources, 3 imbalanced
classes, ~20-feature X-ray/optical/reliability feature set), the
classification problem saturates almost immediately relative to the
available label budget - random sampling reaches ~90% of its final
macro-F1 within ~150 of 1,753 pool labels. There simply isn't much room
for a smarter query strategy to outperform random when the pool is this
small and the useful signal this concentrated; AL's advantage in the
literature (e.g. RB-C1000) shows up in much larger pools where random
sampling wastes most of its budget on redundant examples before reaching
the informative region. This is a legitimate, if unglamorous, negative
result for AL's applicability to counterpart-classification-scale CSC
label budgets, and the shared `common/` infrastructure it was built on
is unaffected - it is exactly what the later two modules will reuse.

Raw per-round results: `results/label_efficiency_log.csv`.
Reproduce: `python -m catalog_classification.run_experiment --seeds 15`.

## Setup

```
pip install -r requirements.txt
python scripts/fetch_data.py
python -m pytest
python -m catalog_classification.run_experiment
```
