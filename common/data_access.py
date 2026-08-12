"""Shared Chandra archive access layer.

Resolves a CSC 2.1 source to its three native data products:

  (a) catalog row      - CSC 2.1 TAP query (works, used by catalog_classification)
  (b) image cutout      - stub for the future image_anomaly_detection module
  (c) event file        - stub for the future eventfile_representation module

All three come off the same archive and the same TAP service (csc21.image
and ivoa.ObsCore carry the cutout/event product locations), so one access
layer with a local disk cache serves all three planned projects rather than
each reimplementing Chandra I/O.

Only (a) is implemented end-to-end here. (b) and (c) raise NotImplementedError
with the query they'll need to issue, so the later modules extend this file
instead of rewriting it.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import pyvo

TAP_URL = "http://cda.cfa.harvard.edu/csc21tap"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _search_with_retry(service, query: str, attempts: int = 4, base_delay: float = 2.0):
    """The public CSC TAP endpoint occasionally drops connections mid-query
    (observed: RemoteDisconnected on large IN(...) queries). Retry with
    exponential backoff before giving up - this is shared plumbing other
    projects will hit the same flakiness through.
    """
    last_exc = None
    for attempt in range(attempts):
        try:
            return service.search(query)
        except Exception as exc:  # pyvo wraps requests/urllib3 errors variously
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc


@dataclass
class ChandraSource:
    """A resolved CSC 2.1 source. Image/event access is lazy and stubbed."""
    name: str
    catalog_row: dict
    _archive: "ChandraArchive" = field(repr=False)

    def image_cutout(self, band: str = "broad", size_arcsec: float = 60.0) -> Path:
        """Download/return a local FITS cutout path for this source.

        Not implemented yet - for image_anomaly_detection to fill in. The
        product lives in csc21.image / ivoa.ObsCore on the same TAP service
        this module already queries; join on `name` (or region/obsid) to
        get the access_url, then cache it under
        `{cache_dir}/{name}/cutout_{band}.fits` following the same
        cache-key convention as query_adql().
        """
        raise NotImplementedError(
            "image cutout access not implemented; see docstring for the "
            "csc21.image / ivoa.ObsCore query this should issue"
        )

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

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, tap_url: str = TAP_URL):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tap_url = tap_url
        self._service: Optional[pyvo.dal.TAPService] = None

    @property
    def service(self) -> pyvo.dal.TAPService:
        if self._service is None:
            self._service = pyvo.dal.TAPService(self._tap_url)
        return self._service

    def _source_cache_path(self, name: str) -> Path:
        safe = name.replace(" ", "_").replace("/", "_")
        d = self.cache_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        return d / "catalog.json"

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
