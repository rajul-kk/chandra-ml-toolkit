"""Setup step 4 (discovery mode): build a seed set from literature-verified
extended X-ray sources instead of a bare extent_flag proxy, and a much
larger, undeduped unlabeled pool for AnomalyMatch to explore.

Why this exists (see image_anomaly_detection/PLAN.md "Setup step 4"): the
validated result in build_seed_cutouts.py shows AnomalyMatch can re-derive
extent_flag from pixels alone, which is a real methods-transfer result but
not a scientific finding, since CSC's own pipeline already computed
extent_flag. This script's seed set is instead cross-matched against
SIMBAD's own catalog of known extended X-ray-emitting object classes (SNRs,
galaxy clusters), so a match here means a real, independent, literature-
backed source, not just a CSC-internal statistical flag.

CSC's TAP backend does not support ADQL geometry functions (CONTAINS/CIRCLE
- confirmed live, it errors with a SQL Server-flavored message), so the
cross-match uses a plain ra/dec box filter instead, matching the approach
already used for image_cutout()'s SIA search-radius handling.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pyvo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data_access import ChandraArchive, _with_retry  # noqa: E402
from image_anomaly_detection.build_seed_cutouts import (  # noqa: E402
    MIN_EXTENT_ARCSEC, MIN_SEPARATION_ARCMIN, _dedup_by_separation,
)

SIMBAD_URL = "http://simbad.cds.unistra.fr/simbad/sim-tap"
CROSSMATCH_RADIUS_ARCSEC = 60.0

# Object types whose X-ray emission is expected to be genuinely extended,
# not a point-source proxy: supernova remnants and galaxy-cluster
# intracluster medium are the two best-established classes for this.
SIMBAD_OTYPES = ["SNR", "ClG"]


def fetch_simbad_candidates(otype: str, n: int) -> pd.DataFrame:
    svc = pyvo.dal.TAPService(SIMBAD_URL)
    q = f"SELECT TOP {n} main_id, ra, dec FROM basic WHERE otype='{otype}' AND ra IS NOT NULL"
    return _with_retry(lambda: svc.search(q)).to_table().to_pandas()


def crossmatch_to_csc(archive: ChandraArchive, simbad_df: pd.DataFrame,
                       otype: str) -> pd.DataFrame:
    """For each SIMBAD position, find nearby CSC sources via a box search
    (CSC's TAP backend has no ADQL geometry support), keep only matches
    that are both extent_flag=1 and above the resolvability floor already
    established in build_seed_cutouts.py - a match here means a real
    literature object AND a real, visually resolvable CSC detection, not
    just one or the other.
    """
    radius_deg = CROSSMATCH_RADIUS_ARCSEC / 3600.0
    rows = []
    for i, row in simbad_df.iterrows():
        ra, dec = float(row["ra"]), float(row["dec"])
        dra = radius_deg / max(math.cos(math.radians(dec)), 0.01)
        q = f"""
            SELECT name, ra, dec, extent_flag, significance, major_axis_b
            FROM csc21.master_source
            WHERE ra BETWEEN {ra - dra} AND {ra + dra}
              AND dec BETWEEN {dec - radius_deg} AND {dec + radius_deg}
              AND extent_flag=1 AND conf_flag=0
              AND major_axis_b > {MIN_EXTENT_ARCSEC}
        """
        t = _with_retry(lambda q=q: archive.service.search(q)).to_table().to_pandas()
        for _, r in t.iterrows():
            rows.append({
                "simbad_name": row["main_id"], "simbad_otype": otype,
                "name": r["name"], "ra": r["ra"], "dec": r["dec"],
                "major_axis_b": r["major_axis_b"], "significance": r["significance"],
            })
        if (i + 1) % 25 == 0:
            print(f"  {otype}: {i + 1}/{len(simbad_df)} SIMBAD positions checked, "
                  f"{len(rows)} verified matches so far")
    return pd.DataFrame(rows)


def build_verified_seed_candidates(n_per_otype: int = 150) -> pd.DataFrame:
    archive = ChandraArchive()
    frames = []
    for otype in SIMBAD_OTYPES:
        print(f"Fetching {n_per_otype} SIMBAD '{otype}' positions...")
        simbad_df = fetch_simbad_candidates(otype, n_per_otype)
        print(f"  cross-matching against CSC (box search, {CROSSMATCH_RADIUS_ARCSEC}\" radius)...")
        matched = crossmatch_to_csc(archive, simbad_df, otype)
        print(f"  {len(matched)} verified extended matches from {len(simbad_df)} '{otype}' positions")
        frames.append(matched)
    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset="name")

    # A single literature object (one SNR/cluster) can match several nearby
    # CSC detections (its emission spans multiple detected regions) - same
    # crowded-field risk already diagnosed in setup step 3 (3 of 4 seed
    # examples turning out to be one dense field, which caused an inverted
    # AUROC). Reuse the same spatial dedup so the final seed set spans
    # genuinely distinct objects, not one object counted several times.
    result = result.sort_values("major_axis_b", ascending=False).reset_index(drop=True)
    result = _dedup_by_separation(result, n=len(result), min_sep_arcmin=MIN_SEPARATION_ARCMIN)
    return result


def query_unlabeled_pool(archive: ChandraArchive, n_point: int) -> pd.DataFrame:
    """A much larger, undeduped pool of point-like sources for the model to
    explore during discovery mode. Deliberately NOT spatially deduped like
    the seed set above - the point here is the model seeing the real,
    messy diversity of sources (including ones that happen to share a
    crowded field), not a curated sample. label_idx is 0 for everything
    since none of these are known to be anomalous; the model's own scores
    are what matter, not a ground-truth split.
    """
    q = f"""
        SELECT TOP {n_point} name, ra, dec, significance, flux_aper_b, extent_flag
        FROM csc21.master_source
        WHERE conf_flag=0 AND significance > 10
        ORDER BY significance DESC
    """
    pool = archive.query_adql(q, cache_key=f"discovery_unlabeled_{n_point}")
    pool["label_idx"] = 0
    return pool


def build_unlabeled_pool(n_point: int, out_dir: Path):
    """Download and crop the unlabeled discovery pool, same resumable
    pattern as build_seed_cutouts.build() (the public SIA/TAP endpoints
    drop connections often enough on runs this long that losing all
    progress on one failure isn't acceptable).
    """
    from image_anomaly_detection.build_seed_cutouts import make_cutout_image

    archive = ChandraArchive()
    candidates = query_unlabeled_pool(archive, n_point)
    print(f"{len(candidates)} unlabeled candidate sources")

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    labels_path = out_dir / "labels_unlabeled.csv"
    rows = []
    if labels_path.exists():
        rows = pd.read_csv(labels_path).to_dict("records")
        done_names = {r["source_name"] for r in rows}
        print(f"resuming: {len(rows)} already done")
    else:
        done_names = set()

    for i, row in candidates.iterrows():
        name, ra, dec = row["name"], row["ra"], row["dec"]
        if name in done_names:
            continue
        try:
            src = archive.resolve(name)
            fits_path = src.image_cutout(band="broad")
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        try:
            img = make_cutout_image(fits_path, ra, dec)
        except Exception as e:
            print(f"  skip {name} (crop failed): {e}")
            continue

        filename = f"discovery_{i:06d}.jpeg"
        img.save(images_dir / filename, format="JPEG", quality=95)
        rows.append({"filename": filename, "label": "unlabeled", "label_idx": 0,
                      "split": "train", "source_name": name})
        pd.DataFrame(rows).to_csv(labels_path, index=False)

        # At this pool's composition, unique-field reuse is close to 1:1
        # (confirmed live: 340 downloaded sources needed 325 unique field
        # downloads, 22GB, and would have needed ~190GB to reach 3000 -
        # more than the free disk space available). The shared cache exists
        # to avoid re-downloading a field another source also needs, but
        # that benefit doesn't materialize here, so delete the raw FITS
        # right after cropping instead of letting it accumulate forever.
        fits_path.unlink(missing_ok=True)

        if len(rows) % 25 == 0:
            print(f"  [{len(rows)}/{len(candidates)}] downloaded")

    print(f"\nSaved {len(rows)} unlabeled cutouts to {images_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["seed", "unlabeled"], default="seed")
    p.add_argument("--n_per_otype", type=int, default=150)
    p.add_argument("--n_point", type=int, default=3000)
    p.add_argument("--out_csv", type=str, default="scratch_cutouts/verified_seed_candidates.csv")
    p.add_argument("--out_dir", type=str, default="scratch_cutouts/discovery_pool")
    args = p.parse_args()

    if args.mode == "seed":
        candidates = build_verified_seed_candidates(args.n_per_otype)
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(out_path, index=False)
        print(f"\n{len(candidates)} unique verified seed candidates saved to {out_path}")
        print(candidates[["simbad_name", "simbad_otype", "name", "major_axis_b"]].to_string())
    else:
        build_unlabeled_pool(args.n_point, Path(args.out_dir))
