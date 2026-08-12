"""Tests for common.data_access. TAP calls are faked so these run offline;
a separate live-service check (tests/test_data_access_live.py, opt-in) hits
the real endpoint.
"""
from pathlib import Path

import pandas as pd
import pytest

from common.data_access import ChandraArchive


class FakeResult:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_table(self):
        return self  # stand-in; to_pandas() below closes the loop

    def to_pandas(self):
        return self._df


class FakeService:
    """Records queries issued and returns a canned row per call."""
    def __init__(self, rows_by_query=None, default_df=None):
        self.calls = []
        self.rows_by_query = rows_by_query or {}
        self.default_df = default_df

    def search(self, query):
        self.calls.append(query)
        for needle, df in self.rows_by_query.items():
            if needle in query:
                return FakeResult(df)
        return FakeResult(self.default_df if self.default_df is not None else pd.DataFrame())


@pytest.fixture
def archive(tmp_path):
    return ChandraArchive(cache_dir=tmp_path)


def test_resolve_caches_to_disk_and_avoids_second_network_call(archive):
    df = pd.DataFrame([{"name": "S1", "ra": 1.0, "dec": 2.0}])
    fake = FakeService(default_df=df)
    archive._service = fake

    src1 = archive.resolve("S1")
    assert src1.catalog_row["ra"] == 1.0
    assert len(fake.calls) == 1

    src2 = archive.resolve("S1")  # should hit cache, not call service again
    assert src2.catalog_row["ra"] == 1.0
    assert len(fake.calls) == 1


def test_resolve_raises_keyerror_when_source_not_found(archive):
    fake = FakeService(default_df=pd.DataFrame())
    archive._service = fake
    with pytest.raises(KeyError):
        archive.resolve("does-not-exist")


def test_resolve_force_bypasses_cache(archive):
    df = pd.DataFrame([{"name": "S1", "ra": 1.0}])
    fake = FakeService(default_df=df)
    archive._service = fake
    archive.resolve("S1")
    archive.resolve("S1", force=True)
    assert len(fake.calls) == 2


def test_image_cutout_and_event_file_are_explicit_stubs(archive):
    df = pd.DataFrame([{"name": "S1", "ra": 1.0}])
    fake = FakeService(default_df=df)
    archive._service = fake
    src = archive.resolve("S1")
    with pytest.raises(NotImplementedError):
        src.image_cutout()
    with pytest.raises(NotImplementedError):
        src.event_file()


def test_query_features_by_name_chunks_and_caches(archive):
    calls = []

    class ChunkingFakeService:
        def search(self, query):
            calls.append(query)
            # return one row per name mentioned in the IN(...) clause
            n_names = query.count("'") // 2
            df = pd.DataFrame([{"name": f"S{i}"} for i in range(n_names)])
            return FakeResult(df)

    archive._service = ChunkingFakeService()
    names = [f"S{i}" for i in range(5)]
    result = archive.query_features_by_name(names, columns=["name"], chunk_size=2)
    assert len(calls) == 3  # 2 + 2 + 1
    assert len(result) == 5

    # second call with same names should be a pure cache hit, no new queries
    result2 = archive.query_features_by_name(names, columns=["name"], chunk_size=2)
    assert len(calls) == 3
    assert len(result2) == 5
