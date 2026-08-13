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

**Results (15 seeds, LightGBM): aggregate macro-F1 is flat, but that
average hides a real, class-imbalanced effect.**

![label efficiency curve](results/label_efficiency_curve.png)

At the aggregate level, active learning does **not** produce a
statistically significant macro-F1 saving over random sampling: all
four strategies converge to the same plateau (~0.637-0.643) by
150-300 labels (paired comparison at the plateau, n=75 pooled samples
per strategy: p=0.19 uncertainty, p=0.26 margin, p=0.44
reliability-weighted vs. random - none below 0.05). That number alone
is well below the ~15-20% saving the original scoping treated as a
positive result. (An early 5-seed read reported 80%+ "savings" -
an artifact of matching random's noisy, already-flat final round
rather than a tail-window plateau; see `common/eval_utils.py`'s
`labels_to_match`, fixed once the artifact was caught.)

**But the aggregate hides where the effect actually is.** Breaking
macro-F1 into its three per-class components
(`catalog_classification/analyze_per_class.py`):

![per-class F1 curves](results/per_class_f1_curves.png)

| class | share of pool | random plateau F1 | AL plateau F1 | p (vs. random) |
|---|---|---|---|---|
| AGN | 45% | 0.874 | 0.886 (uncertainty) | **p < 0.0001** |
| STAR | 51% | 0.903 | 0.912 (uncertainty) | **p < 0.0001** |
| COMPACT_OBJECT | 3% | 0.136 | 0.124-0.130 | p = 0.29-0.65 (n.s.) |

Active learning gives a small but highly significant, consistent boost
on the two classes that make up 97% of the pool (AGN, STAR) - real
signal, not noise, at n=75 pooled samples per comparison. On the rare
COMPACT_OBJECT class it gives *no* benefit, and if anything trends
slightly worse than random, though that direction alone isn't
significant. Since macro-F1 weights all three classes equally, a real
~1.2-point gain on two classes gets diluted by a flat-to-negative third
class, netting the small, non-significant aggregate move that looked
like a clean null result before this breakdown.

**Why:** uncertainty-based acquisition scores every pool example by
distance from the classifier's decision boundary. With AGN+STAR at 97%
of the pool, most boundary-adjacent (uncertain) examples are AGN/STAR
confusions, so the acquisition function spends its budget refining that
boundary - which is exactly where it helps. The ~60-source
COMPACT_OBJECT class is numerically too small to dominate uncertainty
rankings on its own, so plain uncertainty sampling doesn't preferentially
resolve it. This is a distinct mechanism from the label-noise hypothesis
`reliability_weighted` was built to test.

**Layer 2 (the noise-correlation hypothesis specifically):** the
mechanism motivating reliability-weighted acquisition - that classifier
uncertainty and counterpart-match unreliability are correlated, so
vanilla uncertainty sampling preferentially queries untrustworthy labels
- does hold at the *class* level (COMPACT_OBJECT's mean `p_match_ind`
is 0.82 vs. AGN's 0.95) but is essentially absent at the *acquisition*
level: the point-biserial correlation between "queried by uncertainty
sampling" and match reliability is r=-0.03 (p=0.19,
`catalog_classification/analyze_reliability_correlation.py`), not
distinguishable from zero. That's consistent with
`reliability_weighted` showing no benefit over plain uncertainty
sampling anywhere in the per-class breakdown either - there's little
acquisition-level noise correlation for it to correct, and the actual
limiting factor (class imbalance in the acquisition function, not label
trustworthiness) is a different problem than the one it was designed
to solve.

**Tried the obvious fix - it didn't work, and why is itself informative.**
The natural response to "uncertainty sampling is crowded out by the
majority classes" is class-balanced acquisition: rank uncertainty
*within* each predicted class rather than globally, so a numerically
tiny class can't be buried under majority-class boundary cases
(`class_balanced_uncertainty_score` in `common/active_learning.py`).
Ran it - same 15-seed protocol, added as a fifth strategy. It changed
nothing: COMPACT_OBJECT plateau F1 = 0.134 vs. random's 0.136 (p=0.87,
the least significant of any strategy tested), while still keeping the
AGN/STAR gains (p<0.0001 both, same as plain uncertainty).

Checked why directly (`analyze_class_composition.py`,
`catalog_classification/analyze_class_composition.py`): class-balanced
acquisition actually queried *fewer* true COMPACT_OBJECT examples over
a full run (33/800) than plain uncertainty did (39/800) - the fix
regressed on its own target metric. The root cause is upstream of
acquisition: with only 10 seed labels for a class this rare, the
classifier's own COMPACT_OBJECT probability is barely discriminative
from the start - mean P(COMPACT_OBJECT) is 0.357 for true compact
objects vs. 0.303 for everything else, almost the same signal. Bucketing
pool examples by *predicted* class (what class_balanced_uncertainty_score
does) inherits that noise: the "predicted-COMPACT_OBJECT" bucket is
mostly not-actually-compact-objects this early on, so ranking within it
surfaces noise, not real candidates. This is a cold-start problem
specific to the rare class, not a batch-composition problem - any
acquisition function built from this classifier's probabilities inherits
the same blindness, whether it looks at global uncertainty or
per-predicted-class uncertainty. Fixing it would need to happen upstream
of acquisition entirely - e.g. class-weighted training so probability
estimates are calibrated for the rare class before they're used to score
anything, or a much larger rare-class seed set than 10 - not a smarter
query strategy layered on top of an under-informed classifier.

**Bottom line:** active learning delivers a real, reproducible,
statistically significant benefit on this pool's majority classes, no
benefit on its minority class, and the reason is diagnosable rather than
mysterious - a cold-start problem in the classifier's own probability
estimates that no acquisition-layer fix can route around. This is well
below the ~15-20% kill-condition threshold as an aggregate label-saving
result, but it's a complete, actionable, mechanistically-explained
finding rather than an unexplained null, and the shared `common/`
infrastructure it was built on (including the now-tested
`class_balanced_uncertainty_score` strategy) is unaffected - it is
exactly what the later two modules will reuse.

Raw per-round results: `results/label_efficiency_log.csv`.
Reproduce: `python -m catalog_classification.run_experiment --seeds 15`,
then `analyze_per_class.py` and `analyze_class_composition.py`.

## Setup

```
pip install -r requirements.txt
python scripts/fetch_data.py
python -m pytest
python -m catalog_classification.run_experiment
```
