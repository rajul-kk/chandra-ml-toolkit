# Project: extend AnomalyMatch to a new archive

Status: **in progress. Setup step 1 (pipeline validation) done. Setup step 2
(real Chandra cutout access) done. Setup step 3 (first real benchmark run)
done and produced a genuine negative/chance-level result with a diagnosed
cause and an applied fix, not yet re-verified - see "Setup step 3" below
before treating this as resolved.**

## Setup step 3 status (2026-08-20): first real benchmark run - chance-level result, diagnosed

Ran `anomalymatch_chandra_kaggle.ipynb` on Kaggle (T4 x2) against the 474-cutout
pool (36 extended / 438 point, unfiltered `extent_flag`). Pipeline ran
end-to-end without crashing (after fixing a real bug - see below) and
produced a **chance-level result**: `final_auroc=0.498`, `baseline_auroc=0.445`
(worse than random pre-training), `improvement_auprc=-0.023` (AUPRC got
*worse* across training cycles). Top-1% precision (25%) looked closer to the
GalaxyMNIST reference (29.3%) but is not meaningful at this pool size (~4-5
images in the top-1% bucket - one lucky/unlucky hit swings it ~20 points).
Reported to the user as an honest negative finding, not reframed around the
one favorable-looking number, per this project's own stated discipline.

**Notebook bug found and fixed along the way**: `anomaly_match`'s own
dataset loader (`AnomalyDetectionDataset`) scans `cfg.data_dir` as a plain
folder of loose image files (`get_image_names_from_folder`) - it does NOT
read the HDF5 for that count. GalaxyMNIST's own prep script writes both a
loose-file folder (`save_images_to_folder`) and the HDF5
(`create_hdf5_file`); the Chandra notebook's data-packing cell only wrote
the HDF5, so `anomaly_match` found "0 total images" despite a correctly-built
474-image HDF5. Fixed by also copying the built JPEGs into `data_dir`.

**Diagnosed root cause of the chance-level result**: cross-checked
`major_axis_b` (CSC's fitted angular size) against the images' native pixel
scale (~0.49 arcsec/px, measured live) for the actual 474-source pool used.
**50% of the `extent_flag=1` ("anomaly") sources had `major_axis_b` below one
native pixel** - a statistically significant extent per CSC's own fitting
algorithm, but literally sub-pixel and invisible in the rendered image.
Median `major_axis_b` was 0.42" (extended) vs 0.27" (point) - a real but tiny
difference, well below what's resolvable at this cutout's pixel scale. This
is not a data-volume or seed-count problem (more training cycles can't teach
a classifier to see a size difference that isn't in the pixels) - it's a
proxy-label quality problem.

**Applied fix**: added `MIN_EXTENT_ARCSEC = 1.0` (~2 native pixels) filter to
`build_seed_cutouts.py`'s extended-source query, ordered by `major_axis_b
DESC` (largest/most visually resolvable extended sources first, not just
most statistically significant). Rebuilt pool (`seed_pool_v4`): 458 cutouts
(438 point / 20 extended - fewer extended sources survived the stricter
filter + image-availability check). A quick manual visual check of a few
examples was inconclusive (one extended cutout was mostly swamped by the
streak artifact, another didn't look obviously larger than a point-source
example) - **not re-verified on Kaggle yet**. Since the notebook imports
`build_seed_cutouts.py` directly rather than duplicating its logic, this fix
applies automatically on the next notebook run, no notebook changes needed.

**Open question for the next run**: does AUROC actually move off ~0.5 with
the filtered pool? If not, the streak artifact or per-image adaptive
normalization (each cutout independently percentile-stretched to its own
min/max, which may partly erase absolute-size information a size-invariant
stretch would preserve) becomes the next suspect - not yet investigated.

## Setup step 1/2 summary

## Setup step 2 status (2026-08-18): real Chandra cutout pipeline

`common/data_access.py`'s `image_cutout()` now downloads real CSC images via
the SIA endpoint (`csc21siap/queryImages`), verified live. Two non-obvious
findings from getting this working, worth keeping so they aren't
re-discovered:

- **SIA search radius must be decoupled from the desired cutout size.** A
  search box matching a small requested cutout (e.g. 60 arcsec) frequently
  finds zero images even for sources with real image products, because a
  single ACIS CCD field of view is ~0.3deg and the SIA search box must be at
  least that large to reliably intersect an observation's footprint. Fixed
  with a 0.3deg search-radius floor, independent of the eventual crop size.
- **CSC's per-observation `regimg` product ("image around source region")
  is aperture-sized, not a fixed postage stamp** - for a point source it can
  be as small as 5x5 pixels, useless for morphology. The full-field
  `ecorrimg`/`img` product (the one SIA's `accref` links to directly) is the
  right choice; crop it yourself via WCS (`astropy.nddata.Cutout2D`) around
  the source RA/Dec instead.
- **CSC's real per-detection file API is undocumented publicly but
  reverse-engineerable from CIAO's `search_csc` open-source implementation**
  (`ciao_contrib/cda/csccli.py` on GitHub): `GET
  https://cda.cfa.harvard.edu/csccli/browse?packageset={obsid}.{obi}.{region_id}/{filetype}/{band}`
  returns the real filename as JSON (region_id empty string for obi-level
  products like `expmap`), which is then passed to
  `csccli/retrieveFile?filename=...&filetype=...&version=rel2.1` to get the
  bytes. `region_id` for a given source+obsid comes from joining
  `csc21.master_stack_assoc` -> `csc21.stack_observation_assoc` (not a
  simple column on `master_source`).
- **The downloaded image has a real, non-trivial instrumental artifact**:
  ACIS frame-transfer "streak" effects (a well-known Chandra CCD readout
  artifact from bright/saturated sources) show up as a diagonal line across
  the field plus sharp, few-pixel-wide negative-value spikes right next to
  bright sources (background-subtraction over-correction at the streak).
  Confirmed this is NOT an exposure-map/low-exposure artifact (exposure map
  is smooth and high right through the defect) and NOT fixable by a small
  median filter alone (the defect is multi-pixel-wide, not isolated salt
  noise). Properly removing it needs CIAO's real streak-masking tools
  (`acis_streak_map`), out of scope for this pyvo-based layer. Decision:
  **accept it as representative real-world imaging noise** rather than
  chase a full CIAO install - it appears independent of the extent_flag
  label (visible in both extended and point-source cutouts equally), so it
  shouldn't create a spurious shortcut for the classifier, and AnomalyMatch's
  own paper validates against real imaging defects (cosmic rays, diffraction
  spikes) in Hubble/JWST/Euclid data anyway.

`image_anomaly_detection/build_seed_cutouts.py` builds a labeled seed pool
from real CSC sources: `extent_flag=1` (extended) as the anomaly-candidate
class, `extent_flag=0` (point-like) as normal - both filtered to
`conf_flag=0 AND significance>10` to avoid marginal detections. Output
matches `prepare_datasets.py`'s format (RGB JPEGs + labels.csv). Validated
at n=30 requested (24 succeeded, 6 skipped where no broad-band image matched
within the search floor - acceptable at this scale): 12 extended / 12 point,
balanced. Each source's full-frame image is ~50-90MB and there's no
per-source dedup for images sharing a field yet, so scaling this up is a
real bandwidth/time decision, not just a parameter bump - worth checking in
before jumping straight to hundreds/thousands.

**Next**: either scale the seed pool up further and get it through
AnomalyMatch on GPU (Kaggle), or treat 24 as enough for an initial
mechanics/separability check first before investing more download time.

## Setup step 1 status (2026-08-16)

Installed AnomalyMatch (ESA GitHub, MIT licensed) into an isolated Python
3.12 venv at `image_anomaly_detection/.venv` (this machine only had Python
3.10; AnomalyMatch requires >=3.11). Vendored clone lives at
`image_anomaly_detection/vendor/AnomalyMatch/` (gitignored - external repo
with its own git history, not tracked here). Prepared GalaxyMNIST (10,000
images, both 96px and 224px) via the repo's own `paper_scripts/prepare_datasets.py`.

**What's confirmed working end-to-end, CPU-only, on this machine:** one
complete active-learning cycle - session init, baseline evaluation over
the full 10,000-image pool (~13 min), FixMatch training for 10 iterations,
model checkpointing (`.safetensors`), prediction rescoring, and label
correction based on results. This validates the pipeline mechanically
works, matching the plan's Setup step 1 goal.

**What doesn't work reliably: a second training cycle within the same
process.** It crashes with a native `Windows fatal exception: access
violation`, always at the second cycle's very first forward pass, but in
a *different* torch CPU op each time (first `conv2d`, then `hardtanh`
after disabling MKL-DNN) - a pattern consistent with memory corruption
from something in the between-cycles state (model reload, optimizer/EMA
reset, or thread-pool teardown) rather than a bug in any single op. Tried
the standard remedies for this crash class: `num_workers=0` (ruled out
DataLoader multiprocessing), single-threaded MKL/OpenMP (delayed the
crash from cycle 1 to cycle 2, didn't fix it), and
`torch.backends.mkldnn.enabled=False` (changed which op crashes, didn't
fix it). Stopped there rather than continuing to chase CPU-specific
workarounds - both crashing ops are CPU-kernel-specific (oneDNN CPU
conv, CPU hardtanh); GPU execution uses an entirely different code path
(cuDNN) and very plausibly doesn't hit this at all, consistent with the
plan's original "i7 laptop + Colab/Kaggle GPU" hardware assumption.

**Environment patches made to the vendored `paper_scripts/` (all via env
vars with the original GPU-tuned defaults preserved, so nothing changes
for a real GPU run):**
- `ANOMALYMATCH_PRED_BATCH_SIZE` (default 1000) - 1000 OOM'd on this
  machine's ~5GB free RAM; use 32 for CPU.
- `ANOMALYMATCH_NUM_WORKERS` (default 4) - use 0 on CPU to avoid spawning
  worker subprocesses.
- Fixed a real bug in the vendored script unrelated to CPU/GPU: it still
  hardcoded `.pth` checkpoint filenames from before `anomaly_match`
  v1.3.1's migration to `.safetensors` (noted in that package's own
  README changelog) - now uses `.safetensors` throughout.
- TurboJPEG's native lib isn't present on this Windows machine; guarded
  the module-level `TurboJPEG()` instantiation so it falls back to the
  PIL decode path the script already had.

**Recommendation:** proceed to Chandra CSC cutout adaptation work now
(archive access, schema mapping, Cutana/fitsbolt normalisation for CSC's
FITS format) - none of that needs a GPU. Defer actual FixMatch training
runs to Colab/Kaggle GPU, consistent with the original plan, rather than
continuing to debug this CPU-specific crash locally.

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
