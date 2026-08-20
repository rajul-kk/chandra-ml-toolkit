"""Build a small real-data seed set of Chandra CSC image cutouts for
AnomalyMatch, using extent_flag as a cheap proxy label: extended sources
(extent_flag=1) as the anomaly class, point-like sources (extent_flag=0)
as normal - real morphology, not synthetic.

This is a validation-scale run (tens of sources, not the full pool): each
source's downloaded image is a full CCD frame (~50-100MB), so scaling this
to hundreds/thousands of sources is a real bandwidth/disk decision, not
just a parameter change - deliberately kept small here to confirm the
query -> download -> crop -> normalize -> save chain is correct on real
data before committing to that cost.

Output matches the format `paper_scripts/prepare_datasets.py` produces
(RGB JPEGs + a labels.csv with filename/label/label_idx/split columns),
so it's a drop-in swap for AnomalyMatch's benchmark scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import AsinhStretch, PercentileInterval
from astropy.wcs import WCS
from PIL import Image
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data_access import ChandraArchive  # noqa: E402

CLASS_NAMES = {0: "point", 1: "extended"}
CUTOUT_SIZE_ARCSEC = 90.0  # ~1.5 arcmin - a few times a typical PSF, room for extended morphology
IMG_SIZE = 224


# Downloaded ecorrimg cutouts have a native pixel scale of ~0.49 arcsec/px
# (measured live). A validation run found 50%+ of extent_flag=1 sources had
# major_axis_b below this - a statistically significant extent per CSC's own
# fit, but literally sub-pixel and invisible in the image we feed the model.
# That's not a data-volume problem no amount of training fixes; filter to
# sources with a real chance of being visually resolvable instead.
MIN_EXTENT_ARCSEC = 1.0  # ~2 native pixels


def query_candidates(archive: ChandraArchive, n_extended: int, n_point: int) -> pd.DataFrame:
    """Pull real CSC sources by extent_flag, filtered to decent-quality
    detections (avoid marginal significance/confused-field sources that
    would just add label noise to a first validation pass)."""
    q_extended = f"""
        SELECT TOP {n_extended} name, ra, dec, significance, flux_aper_b, extent_flag, major_axis_b
        FROM csc21.master_source
        WHERE extent_flag=1 AND conf_flag=0 AND significance > 10
              AND major_axis_b > {MIN_EXTENT_ARCSEC}
        ORDER BY major_axis_b DESC
    """
    q_point = f"""
        SELECT TOP {n_point} name, ra, dec, significance, flux_aper_b, extent_flag
        FROM csc21.master_source
        WHERE extent_flag=0 AND conf_flag=0 AND significance > 10
        ORDER BY significance DESC
    """
    extended = archive.query_adql(q_extended, cache_key=f"seed_extended_v2_minext{MIN_EXTENT_ARCSEC}_{n_extended}")
    point = archive.query_adql(q_point, cache_key=f"seed_point_{n_point}")
    extended["label_idx"] = 1
    point["label_idx"] = 0
    return pd.concat([extended, point], ignore_index=True)


def make_cutout_image(fits_path: Path, ra: float, dec: float,
                       size_arcsec: float = CUTOUT_SIZE_ARCSEC) -> Image.Image:
    """Crop the downloaded full-frame FITS to a small cutout centered on
    (ra, dec) via WCS, then asinh-stretch to an 8-bit RGB JPEG-ready image
    (asinh rather than linear because CSC's exposure-corrected images are
    near-zero-flux with a long positive tail - linear scaling makes real
    structure invisible)."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header).celestial
        pixscale_deg = abs(float(header["CDELT1"]))

    from astropy.coordinates import SkyCoord
    import astropy.units as u

    center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    size_px = int(size_arcsec / 3600.0 / pixscale_deg)
    cutout = Cutout2D(data, position=center, size=size_px, wcs=wcs, mode="partial",
                       fill_value=0.0)

    arr = np.nan_to_num(cutout.data, nan=0.0)
    # Real CSC ecorrimg products carry ACIS frame-transfer "streak" artifacts
    # (background-subtraction over-correction near bright sources shows up as
    # sharp few-pixel-wide negative spikes) - a 3x3 median filter suppresses
    # these without erasing genuine smooth source structure. Confirmed via
    # direct pixel inspection this is a real instrumental effect (exposure
    # map is smooth/high right through the defect), not fixable by exposure
    # masking; full removal needs CIAO's streak tools, out of scope here -
    # accepted as representative real-world imaging noise.
    arr = median_filter(arr, size=3)
    interval = PercentileInterval(99.5)
    vmin, vmax = interval.get_limits(arr)
    stretch = AsinhStretch(a=0.1)
    normed = stretch((np.clip(arr, vmin, vmax) - vmin) / max(vmax - vmin, 1e-12))
    img_u8 = (np.clip(normed, 0, 1) * 255).astype(np.uint8)

    img = Image.fromarray(img_u8, mode="L").convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return img


def build(n_extended: int, n_point: int, out_dir: Path):
    archive = ChandraArchive()
    candidates = query_candidates(archive, n_extended, n_point)
    print(f"{len(candidates)} candidate sources ({n_extended} extended, {n_point} point)")

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    labels_path = out_dir / "labels_chandra.csv"
    rows = []
    if labels_path.exists():
        # resume: skip sources already downloaded by a prior (possibly
        # interrupted) run of this same out_dir - the public SIA/TAP
        # endpoints drop connections often enough on runs this long that
        # losing all progress on one failure isn't acceptable.
        rows = pd.read_csv(labels_path).to_dict("records")
        done_names = {r["source_name"] for r in rows}
        print(f"resuming: {len(rows)} already done")
    else:
        done_names = set()

    for i, row in candidates.iterrows():
        name, ra, dec, label_idx = row["name"], row["ra"], row["dec"], int(row["label_idx"])
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

        filename = f"chandra_{i:05d}_{label_idx}.jpeg"
        img.save(images_dir / filename, format="JPEG", quality=95)
        rows.append({"filename": filename, "label": CLASS_NAMES[label_idx],
                      "label_idx": label_idx, "split": "train", "source_name": name})
        pd.DataFrame(rows).to_csv(labels_path, index=False)  # write after each success, not just at the end
        print(f"  [{len(rows)}/{len(candidates)}] {name} -> {filename} ({CLASS_NAMES[label_idx]})")

    labels_df = pd.DataFrame(rows)
    print(f"\nSaved {len(labels_df)} cutouts to {images_dir}")
    print(f"Labels: {labels_df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_extended", type=int, default=10)
    p.add_argument("--n_point", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="scratch_cutouts/seed_set")
    args = p.parse_args()
    build(args.n_extended, args.n_point, Path(args.out_dir))
