# Project 3 (queued): extend E-t Map / E-t-dt Cube representation learning to a new archive

Status: **queued, not started.** Start after `catalog_classification/` (Project 1)
is fully wrapped up, per the repo's sequencing rule (don't run modules in
parallel - get one clean result first).

## Source method

Dillmann & Martinez-Galarza et al., "Representation Learning for Time-Domain
High-Energy Astrophysics: Discovery of Extragalactic Fast X-ray Transient
XRT 200515" (MNRAS 537, 931, 2025; arXiv:2412.01150).

- Two fixed-length representations for variable-length X-ray event-file time
  series (raw photon time/energy lists, not images): **E-t Maps** and
  **E-t-dt Cubes**.
- Feature extraction via PCA or a sparse autoencoder.
- Clustering via DBSCAN.
- Nearest-neighbor search around known transients to surface new candidates.
- Applied to a Chandra subset: 3,559 transient candidates, one genuine
  discovery (XRT 200515, an extragalactic fast X-ray transient).
- Paper explicitly states the method "extends to data sets from other
  observatories such as XMM-Newton, Swift-XRT, eROSITA, Einstein Probe, and
  upcoming missions like AXIS" - not yet demonstrated on any of them as of
  the last check (August 2026).

## Mandatory first step when this project starts

**Re-run the novelty check before writing any code** - search "E-t map" OR
"E-t-dt cube" + each candidate archive name (XMM-Newton, Swift-XRT, eROSITA,
Einstein Probe), and check the first author's more recent papers. This gap
is worth re-verifying every time this project is picked up, not just once,
since it can close between when this was written down and when work
actually starts.

## Setup (in order)

1. **Read the source paper in full.** Check its data-availability statement
   and any linked GitHub for public code/data. If code is public, reproduce
   the Chandra pipeline on a small subset first - confirm E-t Map /
   E-t-dt Cube construction and the autoencoder+DBSCAN pipeline actually
   works before touching new data. This is the same "verify the premise on
   a small real slice before scaling" discipline that worked well for
   Project 1's CSC/label-source verification.
2. **Pick ONE target archive**, based on public event-file accessibility:
   - **eROSITA DR1/DR2** - event files may require registration/HEASARC
     access; check current terms first. Highest scientific interest (newer,
     less-studied source population).
   - **XMM-Newton** - event files public via the XMM-Newton Science Archive
     (XSA). Likely the most straightforward public access; large archive;
     event file format similar to Chandra's.
   - **Swift-XRT** - event files public via the UK Swift Science Data
     Centre. Smaller individual exposures; may need adaptation for
     shorter/sparser event files.
3. **Adapt the E-t Map / E-t-dt Cube construction** to the chosen archive's
   event-file format. FITS event lists differ in column names and
   calibration between missions - don't assume Chandra's schema transfers
   unchanged. Verify column-by-column against the target archive's actual
   files, not just the format documentation.
4. **Run the same PCA/autoencoder + DBSCAN + nearest-neighbor pipeline.**
   Cross-check any high-confidence transient candidates against
   SIMBAD/NED/the Transient Name Server before claiming novelty.
5. **Report as a methods contribution even without a discovery**: does the
   representation transfer cleanly, or does it need adaptation? Which
   failure modes appear on the new archive's noise/PSF/exposure
   characteristics? (This framing already anticipates a Project-1-style
   negative/mixed result being a legitimate, publishable outcome - keep
   that honesty standard here too.)

## Kill condition

If the source authors' code isn't public and reimplementation from the
paper's description alone proves unreliable after a focused first attempt,
don't sink further time reverse-engineering it - pivot to a different queued
project rather than grinding on an underspecified reimplementation.

## Repo location

This project lives inside `chandra-toolkit` at `eventfile_representation/`,
reusing `common/data_access.py` (extend the currently-stubbed `event_file()`
method for the chosen archive) and, where useful, `common/eval_utils.py`'s
logging conventions - it does not need `common/active_learning.py` since
this is unsupervised representation learning + clustering, not a labeling
loop.
