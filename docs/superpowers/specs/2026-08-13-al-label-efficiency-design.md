# chandra-ml-toolkit: active-learning label efficiency (design)

Date: 2026-08-13

## Goal

Build `chandra-ml-toolkit`, a repo eventually holding three projects that
attack the Chandra archive through its three native data products (catalog
features, image cutouts, event files). This spec covers the first project
- active-learning label efficiency for CSC 2.1 source classification - plus
the shared `common/` modules it establishes for the other two to reuse.

## Repo structure

```
chandra-ml-toolkit/
├── common/
│   ├── data_access.py       # Chandra TAP access + disk cache; cutout/event stubs
│   ├── active_learning.py   # domain-agnostic AL loop
│   └── eval_utils.py        # label-efficiency curves + CSV logging
├── catalog_classification/  # this project
├── image_anomaly_detection/ # placeholder
├── eventfile_representation/# placeholder
└── README.md
```

## Data

- **Catalog features:** CSC 2.1 via TAP (`http://cda.cfa.harvard.edu/csc21tap`,
  tables `csc21.master_source`, `csc21.image`, `ivoa.ObsCore`). Confirmed
  working from this environment.
- **Class labels:** MUWCLASS training set (Yang et al. 2022,
  arXiv:2206.13656), SIMBAD-derived, ~2,962 sources, GitHub
  `huiyang-astro/MUWCLASS_CSCv2`.
- **Counterpart reliability:** Chandra-Gaia Catalog of Counterparts
  (Perez-Diaz et al. 2026, arXiv:2606.19329), Zenodo DOI
  10.5281/zenodo.18652667, columns `p_i`/`p_any`/`p_match_ind`.
- **Join:** inner join on CSC name -> 2,191 labeled, reliability-scored
  sources. No expansion path without new labels (MUWCLASS caps at 2,962
  total).

## Label scheme

3-class, not the originally planned 4-class star/AGN/galaxy/compact-object:
MUWCLASS has no "galaxy" label (X-ray point sources are essentially never
resolved galaxies). Real classes collapse to:

- `AGN` <- AGN
- `STAR` <- LM-STAR, HM-STAR, YSO
- `COMPACT_OBJECT` <- NS, NS_BIN, CV, LMXB, HMXB

Class balance: STAR 1128, AGN 988, COMPACT_OBJECT 75 (imbalanced -> macro-F1
is the primary metric, accuracy secondary).

## Novelty check (August 2026)

No Chandra/eROSITA counterpart-classification pipeline applies active
learning; all use fixed pre-curated training sets (checked against
arXiv:2606.19329 and arXiv:2509.02842, both post-dating the original scoping
search).

## Methods contribution (layer 2)

Vanilla uncertainty sampling queries sources nearest the decision boundary.
In this domain those sources are disproportionately ones whose counterpart
match is itself ambiguous (positional/photometric ambiguity drives both
classifier uncertainty and match unreliability) - documented as a real risk
by arXiv:2607.13233 (label noise / luminosity overlap in X-ray AGN/SFG
classification). Measured directly: COMPACT_OBJECT (the hardest, rarest
class) has the lowest mean `p_match_ind` (0.83) vs. AGN (0.95), so this
correlation is present in the actual data, not hypothetical.

Contribution: `reliability_weighted()` in `common/active_learning.py` wraps
any base acquisition score with a per-example reliability multiplier
(`score * reliability**alpha`), giving a query strategy that trades off
informativeness against expected label trustworthiness. Evaluated against
vanilla uncertainty/margin sampling and random control.

## Application layer (layer 1)

- Pool: 2,191-source feature matrix (X-ray + Gaia + reliability columns,
  missing-indicators for physically-missing hardness/variability).
- Estimator: LightGBM (already installed, matches prior CSC/eROSITA papers).
- Stratified 80/20 train/test split; stratified per-class initial seed set
  (else random init can miss the 75-source COMPACT_OBJECT class entirely).
- Strategies compared: random, uncertainty, margin, reliability_weighted.
- Metric: macro-F1 primary (imbalance), accuracy secondary.
- Headline number: label-savings fraction (`eval_utils.label_savings_fraction`)
  vs. random at random's final macro-F1.

## Kill condition

If active learning gives <15-20% label savings over random, report it
honestly as a negative/mixed finding - does not block the shared
infrastructure being reused by the other two modules.

## Testing

Every shared module (`data_access`, `active_learning`, `eval_utils`) and the
project-specific `dataset`/`features` modules have unit tests run offline
against fixtures, plus at least one test that checks real behavior (not just
plumbing) - e.g. `test_uncertainty_sampling_beats_random_on_separable_data...`
in `tests/test_active_learning.py`.

## Sequencing

Only `catalog_classification/` is built out now. `image_anomaly_detection/`
and `eventfile_representation/` stay empty placeholders until this project
has a complete, tested, working result end to end.
