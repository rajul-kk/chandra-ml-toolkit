# Project (queued): extend AnomalyMatch to a new archive

Status: **queued, not started.** Try after Project 1 (`catalog_classification/`)
is fully wrapped up - it currently has one experiment still running (the
classifier-independent `prototype_distance_score` strategy, the last untried
acquisition idea). Don't start this in parallel; get that result and close
Project 1 out first, per the repo's sequencing rule.

## Source method

Gomez et al., AnomalyMatch (arXiv:2505.03509) - a FixMatch semi-supervised
classifier + active learning for anomaly detection. Code public:
github.com/esa/AnomalyMatch.

- Already run by the original ESA team on: full Hubble Legacy Archive
  (99.6M cutouts, arXiv:2505.03508), JWST (57 lens candidates / 600,000
  sources), Euclid Q1 (61 jellyfish galaxies / 380,000 sources).
- **Do NOT propose Hubble, JWST, or Euclid** - already swept.
- ZTF is now **partially contested**: "Anomaly Hunter for Alerts (AHA)"
  (Iskandarli et al., arXiv:2602.12955, Feb 2026) already runs unsupervised
  anomaly detection on the ZTF alert stream via Lasair, using an autoencoder
  ensemble on features/cutouts/light curves separately - a different
  architecture from AnomalyMatch's FixMatch+AL approach, so porting
  AnomalyMatch to ZTF is still a distinct methods comparison, but no longer
  a clean "unswept archive" claim. Requires explicit differentiation from
  AHA, not a first-mover framing.

## Mandatory first step when this project starts

**Re-verify the novelty gap before writing any code.** This field moves
fast (AHA closed part of the ZTF claim in the ~3 months since the original
scoping search) - re-check each candidate archive against AnomalyMatch and
against any equivalent FixMatch+AL or comparable anomaly-detection method,
not just search for the exact phrase "AnomalyMatch."

## Candidate archives, in priority order

1. **Chandra Source Catalog 2.1 imaging cutouts** (highest priority) -
   different wavelength regime (X-ray) and noise/PSF characteristics than
   anything AnomalyMatch has been validated on. Public via the CSC image
   cutout service. No anomaly-detection application of AnomalyMatch or an
   equivalent found here as of August 2026. **This is a genuine Chandra
   data product** - if chosen, this project stays inside `chandra-toolkit`
   as `image_anomaly_detection/`, and is the natural implementer of
   `common/data_access.py`'s currently-stubbed `image_cutout()` method,
   which was written with exactly this future use in mind (see that
   method's docstring for the `csc21.image`/`ivoa.ObsCore` query it
   expects). Also the only candidate that keeps this project inside the
   Chandra archive alongside Projects 1 and 3 (event-file), the three-
   data-product framing the whole repo was originally scoped around.
2. **Spitzer Heritage Archive** - infrared, large legacy archive, minimal
   recent systematic anomaly-search coverage found. Not a Chandra product -
   would need its own standalone repo if chosen (same reasoning as the
   NEOWISE project).
3. **Parkes pulsar archive** - different data type (time-series/dynamic
   spectra, not 2D cutouts); would require reframing AnomalyMatch's
   image-based approach. Flagged as higher-risk/higher-effort. Not a
   Chandra product - standalone repo if chosen.
4. **ZTF alert cutouts** - viable ONLY if framed explicitly as "does
   AnomalyMatch's FixMatch+AL approach outperform AHA's autoencoder
   ensemble on the same ZTF anomaly-detection task" - a direct method
   comparison, not a first-mover claim. Read AHA (arXiv:2602.12955) in
   full before choosing this option. Not a Chandra product - standalone
   repo if chosen.

## Setup (in order)

1. **Install AnomalyMatch** from the ESA GitHub repo, work through the
   StarterNotebook, and reproduce a small-scale result on a public
   benchmark (GalaxyMNIST or miniImageNet, as in the original paper) -
   confirm the pipeline works before touching real data. Same "verify the
   premise on a small real slice before scaling" discipline that worked
   for Project 1's CSC/label-source verification and should apply here too.
2. **Pick an archive based on actual data-access feasibility** on the
   available hardware (i7 laptop + Colab/Kaggle GPU) - confirm you can
   stream or download cutouts before committing to one.
3. **Adapt AnomalyMatch's Cutana/fitsbolt normalization pipeline** to the
   chosen archive's FITS format and filter bands - don't assume schema
   transfers unchanged (same lesson as Project 3's event-file plan: verify
   column-by-column against real files, not just format documentation).
4. **Seed with 5-10 labeled examples** of a target anomaly class, run 2-3
   active-learning cycles as the original paper did, report AUROC/AUPRC
   plus manually-vetted top-1% precision.
5. **Cross-check any genuine candidate anomalies** against SIMBAD/NED
   before claiming novelty.

## Kill condition

If data-access/streaming setup alone eats more than ~1-2 weeks without a
working pipeline, drop to a smaller archive slice rather than abandoning
the project - the methods contribution (porting AnomalyMatch to a new
modality) is valuable even at smaller scale.

## Note on Project 1's outcome, if relevant when this starts

Project 1 found that under severe class imbalance, every acquisition
strategy tested that depends on the classifier's own probability
estimates failed to help the rare class - only a classifier-independent,
feature-space-distance signal was untried-but-promising as of when this
was written. If AnomalyMatch's active-learning component runs into a
similar rare-anomaly cold-start problem, that diagnostic playbook
(check calibration first, then batch composition, then whether the
acquisition signal depends on the model at all) transfers directly -
see `catalog_classification`'s README and the `chandra-toolkit-project1-finding`
memory for the full chain.
