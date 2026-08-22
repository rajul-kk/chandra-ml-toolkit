# Project: extend AnomalyMatch to a new archive

Status: first genuine positive result reached (2026-08-22). AnomalyMatch
shows a real, monotonically improving discrimination signal on real Chandra
CSC cutouts (final_auroc 0.715, beating the GalaxyMNIST validation
reference of 0.627), after two rounds of diagnosing and fixing proxy-label
problems. This is a pipeline and methods-transfer validation result, not a
discovery claim. See "Setup step 3, third run" below for the full picture
and caveats. Step 4 (discovery mode) is scoped at the bottom of this file
and is now in progress.

## RESULT (2026-08-22): AnomalyMatch transfers to Chandra X-ray imaging

Headline number: final_auroc = 0.715 on real CSC 2.1 image cutouts, using
extent_flag (extended vs. point-like) as the anomaly/normal split, on a
471-image pool (33 extended, 438 point), 20-example seed, 2 active-learning
cycles. The run improved monotonically: baseline_auroc 0.370 (below chance
before training), first_iter_auroc 0.628, final_auroc 0.715. final_auprc
0.245 against a 7% base rate (33/471) is about 3.5x the base rate, the most
statistically solid number here. Top-0.1%/top-1% precision (100%/75%) look
excellent but are thin at this pool size (roughly 1 and 4-5 images in those
buckets), so don't over-read them.

Why it took three attempts, briefly (full detail in "Setup step 3" below):
run 1 was chance-level (final_auroc 0.498) because extent_flag included
many sources with sub-pixel measured extent. That's a real statistical
property CSC's own fitting algorithm found, but it's invisible in the
rendered image, so no amount of training could teach the model to see it.
Run 2, after filtering to major_axis_b > 1.0", produced a strong but
inverted signal (final_auroc 0.182, equivalent to about 0.82 flipped)
because 3 of the 4 initial anomaly seed examples turned out to be the same
dense stellar cluster field (source confusion, not real diffuse structure),
which dominated the model's whole anomaly signal. Run 3, after adding
spatial dedup so one crowded field can't supply multiple seed examples,
produced the result above.

What this establishes: AnomalyMatch's FixMatch plus active-learning
approach isn't just mechanically portable to Chandra X-ray data (setup
steps 1 and 2 already showed that). It can find a real, non-trivial visual
signal on this data product, the first time this method has been applied
to Chandra or X-ray imaging at all. What this does not establish: any
specific astrophysical discovery. extent_flag is a real catalog property,
but the model is only shown to recover a signal already known to CSC's own
pipeline. See "Setup step 4" below for the more profound next step this
opens up.

## Setup step 3 status (2026-08-20): first real benchmark run, chance-level result, diagnosed

Ran anomalymatch_chandra_kaggle.ipynb on Kaggle (T4 x2) against the
474-cutout pool (36 extended, 438 point, unfiltered extent_flag). The
pipeline ran end to end without crashing (after fixing a real bug, see
below) and produced a chance-level result: final_auroc 0.498, baseline_auroc
0.445 (worse than random before training), improvement_auprc -0.023 (AUPRC
got worse across training cycles). Top-1% precision (25%) looked closer to
the GalaxyMNIST reference (29.3%) but isn't meaningful at this pool size
(roughly 4-5 images in the top-1% bucket, so one lucky or unlucky hit swings
it about 20 points). Reported to the user as an honest negative finding
rather than reframed around the one favorable-looking number, matching this
project's stated discipline.

Notebook bug found and fixed along the way: anomaly_match's own dataset
loader (AnomalyDetectionDataset) scans cfg.data_dir as a plain folder of
loose image files (get_image_names_from_folder). It doesn't read the HDF5
for that count. GalaxyMNIST's own prep script writes both a loose-file
folder (save_images_to_folder) and the HDF5 (create_hdf5_file); the Chandra
notebook's data-packing cell only wrote the HDF5, so anomaly_match found "0
total images" despite a correctly built 474-image HDF5. Fixed by also
copying the built JPEGs into data_dir.

Diagnosed root cause of the chance-level result: cross-checked major_axis_b
(CSC's fitted angular size) against the images' native pixel scale (about
0.49 arcsec/px, measured live) for the actual 474-source pool used. Half of
the extent_flag=1 ("anomaly") sources had major_axis_b below one native
pixel: a statistically significant extent per CSC's own fitting algorithm,
but literally sub-pixel and invisible in the rendered image. Median
major_axis_b was 0.42" for extended sources vs 0.27" for point sources, a
real but tiny difference, well below what's resolvable at this cutout's
pixel scale. This isn't a data-volume or seed-count problem (more training
cycles can't teach a classifier to see a size difference that isn't in the
pixels); it's a proxy-label quality problem.

Applied fix: added MIN_EXTENT_ARCSEC = 1.0 (about 2 native pixels) to
build_seed_cutouts.py's extended-source query, ordered by major_axis_b DESC
(largest, most visually resolvable extended sources first, not just most
statistically significant). Rebuilt the pool (seed_pool_v4): 458 cutouts
(438 point, 20 extended; fewer extended sources survived the stricter
filter plus image-availability check). A quick manual visual check of a few
examples was inconclusive: one extended cutout was mostly swamped by the
streak artifact, another didn't look obviously larger than a point-source
example. Not re-verified on Kaggle yet. Since the notebook imports
build_seed_cutouts.py directly rather than duplicating its logic, this fix
applies automatically on the next notebook run, with no notebook changes
needed.

Open question for the next run: does AUROC actually move off 0.5 with the
filtered pool? If not, the streak artifact or the per-image adaptive
normalization (each cutout is independently percentile-stretched to its own
min/max, which may partly erase absolute-size information that a
size-invariant stretch would preserve) becomes the next suspect. Not
investigated yet.

## Setup step 3 continued (2026-08-21): second real run, strong but inverted signal, second cause found

The seed_pool_v4 re-run (major_axis_b > 1.0" filter) produced final_auroc
0.182, which is not chance-level. It's a strong, systematically inverted
signal (flip it and you get roughly 0.82). Top-1% precision of 0% confirms
a real ranking, just backwards. Confirmed via the run's own logs that this
wasn't an indexing bug: class counts and seed composition all matched
expectations exactly.

Second root cause found: inspected the actual 4 initial anomaly-seed images
(create_initial_labeled_data's random_state=0 sample is reproducible
locally against the same labels_chandra.csv). 3 of the 4 turned out to be
the same dense stellar cluster field (Carina-region coordinates, a few
arcmin apart), a starfield of many blended point sources rather than a
genuine diffuse blob. extent_flag=1 there almost certainly reflects source
confusion (CSC's detector can't cleanly separate close point sources), not
real large-scale morphology. With only 20 extended sources total and a
4-example seed, one crowded field dominated the model's entire initial
anomaly signal: a narrower, different visual pattern than the diversity of
what actually got labeled extent_flag=1 elsewhere in the pool. That
plausibly explains the inversion; the model learned one field's texture,
which doesn't generalize and may anti-correlate with the rest of the class.

Applied fix: MIN_SEPARATION_ARCMIN = 5.0, a greedy spatial dedup on
extended candidates (_dedup_by_separation) so a single crowded field can't
supply multiple "different" examples. The first attempt combined this with
the existing ORDER BY major_axis_b DESC and collapsed to only 4 unique
extended sources in the entire 600-row fetch. Large, rare extended
structures cluster hard in a few famous regions (Carina, LMC, and so on),
so sorting by size before deduping just finds the same few regions
repeatedly. Fixed by fetching a much larger candidate set (TOP 2000, ORDER
BY significance DESC instead of size) and shuffling (random_state=0) before
the spatial dedup, so the sample spreads across genuinely different sky
regions rather than concentrating on rare giant structures. Rebuilt
(seed_pool_v6): 471 cutouts (438 point, 33 extended), all distinct fields
and sources this time.

Notebook bug also found this round: `!pip install -q . pyvo astropy scipy`
in one command silently failed to install pyvo. Combining the local `.`
package with unrelated pip packages let the resolver drop one silently, and
`-q` hid the evidence. ModuleNotFoundError surfaced three cells later at an
unrelated import site. Fixed by splitting into separate pip install calls
plus a fail-fast `import pyvo` right after, so a future failure here is
immediately obvious instead of confusing.

## Setup step 3, third run (2026-08-22): first genuine positive result

seed_pool_v6 (spatially deduped, size-filtered extended sources) through
the same 2-cycle benchmark protocol produced a real, monotonically
improving signal: baseline_auroc 0.370 (below chance before training),
first_iter_auroc 0.628, final_auroc 0.715. This beats the GalaxyMNIST
validation run's final_auroc of 0.627, despite a roughly 20x smaller pool
(471 vs 10,000): real evidence the FixMatch plus active-learning approach
transfers to Chandra X-ray imaging, not just that it doesn't crash.
final_auprc 0.245 against a 7% base rate (33/471 extended) is about 3.5x
the base rate, the most robust number here. Top-0.1%/top-1% precision
(100%/75%) are directionally excellent but statistically thin at this pool
size (roughly 1 and 4-5 images in those buckets respectively, so a couple
of lucky or unlucky hits would swing them substantially). final_auprc is
the number to trust more.

Caveats, still real: extent_flag remains a proxy label (a real measured
catalog property, not a vetted anomaly ground truth), and the ACIS streak
artifact (see setup step 2) is still present and accepted, not removed.
This is a pipeline and methods-transfer validation result. It's genuinely
the first positive evidence AnomalyMatch works on Chandra data, but not yet
a discovery claim about any specific source.

Next steps, in order: (1) scale the pool further now that the pipeline and
label-quality fixes are validated. More extended examples would make the
top-N precision numbers statistically solid rather than thin. (2)
Cross-check the model's genuinely highest-scoring candidates, not just the
labeled extended sources, against SIMBAD/NED before any novelty claim, per
this project's kill-condition discipline. (3) Consider whether the streak
artifact or per-image adaptive normalization concerns (raised after the
second run) are still worth investigating now that a real signal exists, or
whether that's diminishing returns given the pipeline validation goal is
already met.

## Setup step 2 status (2026-08-18): real Chandra cutout pipeline

common/data_access.py's image_cutout() now downloads real CSC images via
the SIA endpoint (csc21siap/queryImages), verified live. Two non-obvious
findings from getting this working, worth keeping so they aren't
rediscovered:

- SIA search radius must be decoupled from the desired cutout size. A
  search box matching a small requested cutout (say 60 arcsec) frequently
  finds zero images even for sources with real image products, because a
  single ACIS CCD field of view is about 0.3deg, and the SIA search box
  must be at least that large to reliably intersect an observation's
  footprint. Fixed with a 0.3deg search-radius floor, independent of the
  eventual crop size.
- CSC's per-observation regimg product ("image around source region") is
  aperture-sized, not a fixed postage stamp. For a point source it can be
  as small as 5x5 pixels, useless for morphology. The full-field
  ecorrimg/img product (the one SIA's accref links to directly) is the
  right choice; crop it yourself via WCS (astropy.nddata.Cutout2D) around
  the source RA/Dec instead.
- CSC's real per-detection file API is undocumented publicly but
  reverse-engineerable from CIAO's search_csc open-source implementation
  (ciao_contrib/cda/csccli.py on GitHub). A GET to
  `https://cda.cfa.harvard.edu/csccli/browse?packageset={obsid}.{obi}.{region_id}/{filetype}/{band}`
  returns the real filename as JSON (region_id is an empty string for
  obi-level products like expmap), which is then passed to
  `csccli/retrieveFile?filename=...&filetype=...&version=rel2.1` to get the
  bytes. region_id for a given source and obsid comes from joining
  csc21.master_stack_assoc to csc21.stack_observation_assoc, not a simple
  column on master_source.
- The downloaded image has a real, non-trivial instrumental artifact. ACIS
  frame-transfer "streak" effects (a well-known Chandra CCD readout
  artifact from bright or saturated sources) show up as a diagonal line
  across the field plus sharp, few-pixel-wide negative-value spikes right
  next to bright sources (background-subtraction over-correction at the
  streak). Confirmed this isn't an exposure-map or low-exposure artifact
  (the exposure map is smooth and high right through the defect), and it's
  not fixable by a small median filter alone (the defect is multi-pixel
  wide, not isolated salt noise). Properly removing it needs CIAO's real
  streak-masking tools (acis_streak_map), out of scope for this pyvo-based
  layer. Decision: accept it as representative real-world imaging noise
  rather than chase a full CIAO install. It appears independent of the
  extent_flag label (visible in both extended and point-source cutouts
  equally), so it shouldn't create a spurious shortcut for the classifier,
  and AnomalyMatch's own paper validates against real imaging defects
  (cosmic rays, diffraction spikes) in Hubble, JWST, and Euclid data anyway.

image_anomaly_detection/build_seed_cutouts.py builds a labeled seed pool
from real CSC sources: extent_flag=1 (extended) as the anomaly-candidate
class, extent_flag=0 (point-like) as normal, both filtered to
`conf_flag=0 AND significance>10` to avoid marginal detections. Output
matches prepare_datasets.py's format (RGB JPEGs plus labels.csv). Validated
at n=30 requested (24 succeeded, 6 skipped where no broad-band image
matched within the search floor, acceptable at this scale): 12 extended, 12
point, balanced. Each source's full-frame image is about 50-90MB and there
was no per-source dedup for images sharing a field yet at this point, so
scaling this up is a real bandwidth and time decision, not just a parameter
bump. Worth checking in before jumping straight to hundreds or thousands.

Next: either scale the seed pool up further and get it through AnomalyMatch
on GPU (Kaggle), or treat 24 as enough for an initial mechanics and
separability check before investing more download time.

## Setup step 1 status (2026-08-16)

Installed AnomalyMatch (ESA GitHub, MIT licensed) into an isolated Python
3.12 venv at image_anomaly_detection/.venv (this machine only had Python
3.10; AnomalyMatch requires 3.11 or newer). Vendored clone lives at
image_anomaly_detection/vendor/AnomalyMatch/ (gitignored, an external repo
with its own git history, not tracked here). Prepared GalaxyMNIST (10,000
images, both 96px and 224px) via the repo's own
paper_scripts/prepare_datasets.py.

What's confirmed working end to end, CPU-only, on this machine: one
complete active-learning cycle: session init, baseline evaluation over the
full 10,000-image pool (about 13 min), FixMatch training for 10 iterations,
model checkpointing (.safetensors), prediction rescoring, and label
correction based on results. This validates the pipeline mechanically
works, matching the plan's setup step 1 goal.

What doesn't work reliably is a second training cycle within the same
process. It crashes with a native Windows fatal exception (access
violation), always at the second cycle's very first forward pass, but in a
different torch CPU op each time (first conv2d, then hardtanh after
disabling MKL-DNN). That pattern is consistent with memory corruption from
something in the between-cycles state (model reload, optimizer/EMA reset,
or thread-pool teardown) rather than a bug in any single op. Tried the
standard remedies for this crash class: num_workers=0 (ruled out DataLoader
multiprocessing), single-threaded MKL/OpenMP (delayed the crash from cycle
1 to cycle 2, didn't fix it), and torch.backends.mkldnn.enabled=False
(changed which op crashes, didn't fix it). Stopped there rather than
continuing to chase CPU-specific workarounds; both crashing ops are
CPU-kernel-specific (oneDNN CPU conv, CPU hardtanh), and GPU execution uses
an entirely different code path (cuDNN) that very plausibly doesn't hit
this at all, consistent with the plan's original "i7 laptop plus
Colab/Kaggle GPU" hardware assumption.

Environment patches made to the vendored paper_scripts/ (all via env vars
with the original GPU-tuned defaults preserved, so nothing changes for a
real GPU run):

- ANOMALYMATCH_PRED_BATCH_SIZE (default 1000): 1000 OOM'd on this machine's
  roughly 5GB free RAM, so use 32 for CPU.
- ANOMALYMATCH_NUM_WORKERS (default 4): use 0 on CPU to avoid spawning
  worker subprocesses.
- Fixed a real bug in the vendored script unrelated to CPU/GPU: it still
  hardcoded .pth checkpoint filenames from before anomaly_match v1.3.1's
  migration to .safetensors (noted in that package's own README changelog),
  now uses .safetensors throughout.
- TurboJPEG's native lib isn't present on this Windows machine, so the
  module-level TurboJPEG() instantiation is guarded to fall back to the PIL
  decode path the script already had.

Recommendation: proceed to Chandra CSC cutout adaptation work now (archive
access, schema mapping, Cutana/fitsbolt normalisation for CSC's FITS
format), none of which needs a GPU. Defer actual FixMatch training runs to
Colab/Kaggle GPU, consistent with the original plan, rather than continuing
to debug this CPU-specific crash locally.

## Source method

Gomez et al., AnomalyMatch (arXiv:2505.03509), a FixMatch semi-supervised
classifier plus active learning for anomaly detection. Code is public at
github.com/esa/AnomalyMatch.

Already run by the original ESA team on the full Hubble Legacy Archive
(99.6M cutouts, arXiv:2505.03508), JWST (57 lens candidates out of 600,000
sources), and Euclid Q1 (61 jellyfish galaxies out of 380,000 sources). Do
not propose Hubble, JWST, or Euclid; they're already swept. ZTF is now
partially contested: "Anomaly Hunter for Alerts" (AHA, Iskandarli et al.,
arXiv:2602.12955, Feb 2026) already runs unsupervised anomaly detection on
the ZTF alert stream via Lasair, using an autoencoder ensemble on features,
cutouts, and light curves separately. That's a different architecture from
AnomalyMatch's FixMatch plus AL approach, so porting AnomalyMatch to ZTF is
still a distinct methods comparison, but no longer a clean "unswept
archive" claim. It requires explicit differentiation from AHA, not a
first-mover framing.

## Mandatory first step when this project starts

Re-verify the novelty gap before writing any code. This field moves fast
(AHA closed part of the ZTF claim in about the 3 months since the original
scoping search). Re-check each candidate archive against AnomalyMatch and
against any equivalent FixMatch plus AL or comparable anomaly-detection
method, not just search for the exact phrase "AnomalyMatch."

## Candidate archives, in priority order

1. Chandra Source Catalog 2.1 imaging cutouts (highest priority): a
   different wavelength regime (X-ray) and different noise/PSF
   characteristics than anything AnomalyMatch has been validated on. Public
   via the CSC image cutout service. No anomaly-detection application of
   AnomalyMatch or an equivalent found here as of August 2026. This is a
   genuine Chandra data product. If chosen, this project stays inside
   chandra-toolkit as image_anomaly_detection/, and is the natural
   implementer of common/data_access.py's currently stubbed image_cutout()
   method, which was written with exactly this future use in mind (see that
   method's docstring for the csc21.image/ivoa.ObsCore query it expects).
   Also the only candidate that keeps this project inside the Chandra
   archive alongside Projects 1 and 3 (event-file), the three-data-product
   framing the whole repo was originally scoped around.
2. Spitzer Heritage Archive: infrared, a large legacy archive with minimal
   recent systematic anomaly-search coverage found. Not a Chandra product;
   would need its own standalone repo if chosen (same reasoning as the
   NEOWISE project).
3. Parkes pulsar archive: a different data type (time-series/dynamic
   spectra, not 2D cutouts), which would require reframing AnomalyMatch's
   image-based approach. Flagged as higher risk and higher effort. Not a
   Chandra product; standalone repo if chosen.
4. ZTF alert cutouts: viable only if framed explicitly as "does
   AnomalyMatch's FixMatch plus AL approach outperform AHA's autoencoder
   ensemble on the same ZTF anomaly-detection task," a direct method
   comparison, not a first-mover claim. Read AHA (arXiv:2602.12955) in full
   before choosing this option. Not a Chandra product; standalone repo if
   chosen.

## Setup (in order)

1. Install AnomalyMatch from the ESA GitHub repo, work through the
   StarterNotebook, and reproduce a small-scale result on a public
   benchmark (GalaxyMNIST or miniImageNet, as in the original paper) to
   confirm the pipeline works before touching real data. Same "verify the
   premise on a small real slice before scaling" discipline that worked for
   Project 1's CSC/label-source verification, and should apply here too.
2. Pick an archive based on actual data-access feasibility on the available
   hardware (i7 laptop plus Colab/Kaggle GPU); confirm you can stream or
   download cutouts before committing to one.
3. Adapt AnomalyMatch's Cutana/fitsbolt normalization pipeline to the
   chosen archive's FITS format and filter bands. Don't assume schema
   transfers unchanged (same lesson as Project 3's event-file plan: verify
   column by column against real files, not just format documentation).
4. Seed with 5-10 labeled examples of a target anomaly class, run 2-3
   active-learning cycles as the original paper did, and report AUROC/AUPRC
   plus manually vetted top-1% precision.
5. Cross-check any genuine candidate anomalies against SIMBAD/NED before
   claiming novelty.

## Kill condition

If data-access or streaming setup alone eats more than 1-2 weeks without a
working pipeline, drop to a smaller archive slice rather than abandoning
the project. The methods contribution (porting AnomalyMatch to a new
modality) is valuable even at smaller scale.

## Note on Project 1's outcome, if relevant when this starts

Project 1 found that under severe class imbalance, every acquisition
strategy tested that depends on the classifier's own probability estimates
failed to help the rare class. Only a classifier-independent,
feature-space-distance signal was untried but promising as of when this was
written. If AnomalyMatch's active-learning component runs into a similar
rare-anomaly cold-start problem, that diagnostic playbook (check calibration
first, then batch composition, then whether the acquisition signal depends
on the model at all) transfers directly. See catalog_classification's
README and the chandra-toolkit-project1-finding memory for the full chain.

## Setup step 4 (in progress): discovery mode

The RESULT above validates that the method works: extent_flag is
re-derivable from images alone. But it isn't scientifically profound, since
CSC's own pipeline already computed extent_flag, so the model isn't telling
anyone anything new. The more profound next step, matching how
AnomalyMatch's own paper actually used this method on Hubble, JWST, and
Euclid (surfacing specific candidate sources, not just a validation
metric):

1. Seed set: use the same kind of small, verified "known-interesting"
   examples rather than a catalog-flag proxy, for example a handful of CSC
   sources cross-matched to literature-known extended objects (SNRs,
   cluster or group X-ray emission, resolved jets) via SIMBAD/NED, not just
   extent_flag=1. build_seed_cutouts.py's major_axis_b and spatial-dedup
   logic still applies for building this seed set cleanly.
2. Unlabeled pool: scale up substantially from the current roughly 470, to
   thousands of CSC sources spanning a broad significance and flux range,
   not deduped for diversity the way the labeled seed was. The model should
   see the full messy diversity of real point-like sources here, not a
   curated sample. This is the actual bandwidth cost driver; real per-field
   downloads, and dedup by shared field helps but doesn't eliminate it (see
   setup step 2's dedup notes).
3. More active-learning cycles: 3-5 rather than 2, closer to what the paper
   itself used, since each cycle's manual correction step is where real
   signal gets injected.
4. Manual vetting step, the actual point of this phase: for the final
   model's top-N highest-scoring previously unlabeled candidates (not
   sources already known to be extended), inspect the image and query
   SIMBAD/NED by position. A candidate that isn't already catalogued as
   extended or unusual, and doesn't look like a streak/chip-gap artifact or
   crowded-field confusion on inspection, is the actual finding this phase
   is chasing.
5. Honest kill condition: if none of the top candidates survive step 4 as
   genuinely novel (all are either known objects or artifacts), that's a
   legitimate negative result to report as such, the same discipline as
   every other finding in this project, not a failure of the pipeline,
   which the RESULT above already validates works.
