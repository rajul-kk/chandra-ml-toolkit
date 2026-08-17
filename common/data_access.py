"""Shared Chandra archive access layer.

Resolves a CSC 2.1 source to its three native data products:

  (a) catalog row  - CSC 2.1 TAP query (used by catalog_classification)
  (b) image        - CSC SIA query (`csc21siap/queryImages`), used by
                      image_anomaly_detection - returns the full CCD-frame
                      FITS image, not a cropped cutout (see image_cutout()'s
                      docstring)
  (c) event file    - stub for the future eventfile_representation module

(a) and (b) use different services: TAP (`csc21tap`) for catalog/metadata,
SIA (`csc21siap/queryImages`) for images - CSC image products are not
reachable via TAP's ObsCore `access_url` directly. One access layer with a
local disk cache serves all three planned projects rather than each
reimplementing Chandra I/O.

(c) raises NotImplementedError with the query it'll need to issue, so
eventfile_representation extends this file instead of rewriting it.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from urllib.parse import parse_qs, urlparse

import pandas as pd
import pyvo
import requests

TAP_URL = "http://cda.cfa.harvard.edu/csc21tap"
SIA_URL = "http://cda.cfa.harvard.edu/csc21siap/queryImages"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

# CSC image band names, as they appear (suffixed to "CXO_") in SIA search
# results' `band` column - verified live against the SIA endpoint.
BAND_CODES = {"broad": "b", "hard": "h", "medium": "m", "soft": "s", "ultrasoft": "u"}


def _with_retry(call, attempts: int = 4, base_delay: float = 2.0):
    """The public CSC TAP/SIA endpoints occasionally drop connections
    mid-query (observed: RemoteDisconnected on large IN(...) TAP queries and
    on SIA image searches alike). Retry any zero-arg callable with
    exponential backoff before giving up - this is shared plumbing other
    projects/call sites hit the same flakiness through.
    """
    last_exc = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # pyvo wraps requests/urllib3 errors variously
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc


def _search_with_retry(service, query: str, attempts: int = 4, base_delay: float = 2.0):
    return _with_retry(lambda: service.search(query), attempts, base_delay)


@dataclass
class ChandraSource:
    """A resolved CSC 2.1 source. Image/event access is lazy and stubbed."""
    name: str
    catalog_row: dict
    _archive: "ChandraArchive" = field(repr=False)

    def image_cutout(self, band: str = "broad", size_arcsec: float = 60.0,
                      force: bool = False) -> Path:
        """Download and return a local FITS image path for this source.

        Uses the CSC SIA endpoint (`csc21siap/queryImages`), not the TAP
        service - CSC image products aren't fetchable via TAP's ObsCore
        `access_url` directly, SIA's `accref` is the real download link
        (verified live: resolves to `csccli/retrieveFile?...`).

        Note this returns the *full CCD-frame image* SIA matches on
        (typically 2048x2048px, not cropped to `size_arcsec`) - `size_arcsec`
        is not used as the SIA search radius (verified live: a search box
        matching a small requested cutout size often finds zero images even
        though the source has real image products, because a single-CCD ACIS
        field of view is ~0.3deg and the SIA search box must be at least
        that large to reliably intersect an observation's footprint; a fixed
        0.3deg floor is used for the search instead). Cropping to a tight
        cutout around this source's RA/Dec is a separate step (WCS-based,
        via astropy Cutout2D) left to the caller/build script, since the
        right crop size and normalization depend on the downstream model.
        """
        band_code = BAND_CODES.get(band, band)

        ra = float(self.catalog_row["ra"])
        dec = float(self.catalog_row["dec"])
        search_size_deg = max(size_arcsec / 3600.0, 0.3)
        results = _with_retry(
            lambda: self._archive.sia_service.search(pos=(ra, dec), size=search_size_deg)
        )

        match = next(
            (r for r in results if r["band"].lower() == f"cxo_{band_code}"), None
        )
        if match is None:
            raise LookupError(
                f"no {band!r} ({band_code}) image found for {self.name!r} via SIA "
                f"search at ({ra}, {dec}), size={size_arcsec}arcsec"
            )

        # Cache by the remote image's own filename, not by source name -
        # many sources share the same observation/field, so this avoids
        # re-downloading the same ~50-90MB full-frame image once per source.
        url = match.getdataurl()
        remote_filename = parse_qs(urlparse(url).query)["filename"][0]
        cache_path = self._archive._shared_image_cache_path(remote_filename)
        if cache_path.exists() and not force:
            return cache_path

        def _download():
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            return resp.content

        cache_path.write_bytes(_with_retry(_download))
        return cache_path

    def event_file(self, obsid: Optional[int] = None) -> Path:
        """Download/return a local event-file (evt2) path for this source.

        Not implemented yet - for eventfile_representation to fill in.
        Event file locations are resolvable via ivoa.ObsCore
        (dataproduct_type='event') keyed by obs_id, joined from this
        source's observation_source rows. Cache under
        `{cache_dir}/{name}/events_{obsid}.fits`.
        """
        raise NotImplementedError(
            "event file access not implemented; see docstring for the "
            "ivoa.ObsCore query this should issue"
        )


class ChandraArchive:
    """CSC 2.1 access with a local disk cache keyed by source id / query hash."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, tap_url: str = TAP_URL,
                 sia_url: str = SIA_URL):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tap_url = tap_url
        self._sia_url = sia_url
        self._service: Optional[pyvo.dal.TAPService] = None
        self._sia_service: Optional[pyvo.dal.SIAService] = None

    @property
    def service(self) -> pyvo.dal.TAPService:
        if self._service is None:
            self._service = pyvo.dal.TAPService(self._tap_url)
        return self._service

    @property
    def sia_service(self) -> pyvo.dal.SIAService:
        if self._sia_service is None:
            self._sia_service = pyvo.dal.SIAService(self._sia_url)
        return self._sia_service

    def _source_cache_path(self, name: str) -> Path:
        safe = name.replace(" ", "_").replace("/", "_")
        d = self.cache_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        return d / "catalog.json"

    def _shared_image_cache_path(self, remote_filename: str) -> Path:
        d = self.cache_dir / "_images"
        d.mkdir(parents=True, exist_ok=True)
        return d / remote_filename

    def resolve(self, name: str, columns: Optional[list[str]] = None,
                force: bool = False) -> ChandraSource:
        """Resolve a single CSC 2.1 source name to a ChandraSource."""
        cache_path = self._source_cache_path(name)
        if cache_path.exists() and not force:
            row = json.loads(cache_path.read_text())
            return ChandraSource(name=name, catalog_row=row, _archive=self)

        cols = columns or ["name", "ra", "dec", "significance", "flux_aper_b",
                            "hard_hm", "hard_hs", "hard_ms", "var_intra_index_b",
                            "var_inter_index_b", "extent_flag", "conf_flag"]
        q = f"SELECT {', '.join(cols)} FROM csc21.master_source WHERE name = '{name}'"
        t = _search_with_retry(self.service, q).to_table().to_pandas()
        if len(t) == 0:
            raise KeyError(f"no CSC 2.1 source found for name={name!r}")
        row = t.iloc[0].to_dict()
        cache_path.write_text(json.dumps(row, default=str))
        return ChandraSource(name=name, catalog_row=row, _archive=self)

    def query_adql(self, query: str, cache_key: Optional[str] = None,
                    force: bool = False) -> pd.DataFrame:
        """Run an arbitrary ADQL query against CSC 2.1, cached by query hash.

        Bulk pool-building goes through here rather than one resolve() call
        per source, since the label-efficiency study needs thousands of rows
        at once.
        """
        key = cache_key or hashlib.sha256(query.encode()).hexdigest()[:16]
        cache_path = self.cache_dir / "queries" / f"{key}.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not force:
            return pd.read_parquet(cache_path)

        t = _search_with_retry(self.service, query).to_table().to_pandas()
        t.to_parquet(cache_path)
        return t

    def query_features_by_name(self, names: list[str], columns: Optional[list[str]] = None,
                                chunk_size: int = 200,
                                cache_key: Optional[str] = None) -> pd.DataFrame:
        """Bulk-fetch catalog features for a list of source names, chunked
        to stay under TAP query-length limits, and cached as one unit keyed
        on the sorted name list (not per-source) so repeat pool builds are
        a cache hit.
        """
        cols = columns or ["name", "significance", "flux_aper_b", "hard_hm",
                            "hard_hs", "hard_ms", "var_intra_index_b",
                            "var_inter_index_b", "extent_flag", "conf_flag"]
        key = cache_key or hashlib.sha256(
            ("|".join(sorted(names)) + "|" + ",".join(cols)).encode()
        ).hexdigest()[:16]
        cache_path = self.cache_dir / "queries" / f"features_{key}.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        # per-chunk cache under a subdir keyed by this query's hash: if a
        # later chunk fails (the public endpoint occasionally drops mid-run),
        # a rerun resumes instead of re-fetching everything already pulled.
        chunk_dir = self.cache_dir / "queries" / f"features_{key}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        for i in range(0, len(names), chunk_size):
            chunk_path = chunk_dir / f"{i}.parquet"
            if chunk_path.exists():
                frames.append(pd.read_parquet(chunk_path))
                continue
            chunk = names[i:i + chunk_size]
            in_list = ",".join("'" + n.replace("'", "''") + "'" for n in chunk)
            q = f"SELECT {', '.join(cols)} FROM csc21.master_source WHERE name IN ({in_list})"
            t = _search_with_retry(self.service, q).to_table().to_pandas()
            t.to_parquet(chunk_path)
            frames.append(t)

        result = pd.concat(frames, ignore_index=True)
        result.to_parquet(cache_path)
        for f in chunk_dir.glob("*.parquet"):
            f.unlink()
        chunk_dir.rmdir()
        return result
