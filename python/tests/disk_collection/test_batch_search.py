# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for ``alayalite.DiskCollection.batch_search`` /
``batch_search_with_distance``.

The Laser per-engine tests are gated on the runtime probe
``_laser_support.DISK_LASER_SUPPORTED``; everything else is engine-agnostic
because the validation prologue, padding contract, and num_threads
resolution all live above the engine-dispatch boundary.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from _laser_support import DISK_LASER_SUPPORTED  # noqa: E402
from alayalite import DiskCollection, MetricType

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="DiskCollection v1 is POSIX-only",
)

_FLAT_DIM = 16
_VAMANA_DIM = 16
_LASER_FIXTURE_DIM = 128
_LASER_FIXTURE_R = 64
_LASER_FIXTURE_N = 256
_UINT64_MAX = np.iinfo(np.uint64).max


# ---------------------------------------------------------------------------
# Per-engine collection builders
# ---------------------------------------------------------------------------


def _build_flat(tmp_path: Path, n: int = 200, seed: int = 42) -> DiskCollection:
    path = tmp_path / "flat"
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, _FLAT_DIM)).astype(np.float32)
    ids = np.arange(n, dtype=np.uint64)
    col = DiskCollection(str(path), dim=_FLAT_DIM, metric=MetricType.L2, index_type="disk_flat")
    col.add(vecs, ids)
    col.flush()
    return col


def _build_vamana(tmp_path: Path, n: int = 200, seed: int = 43) -> DiskCollection:
    path = tmp_path / "vamana"
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, _VAMANA_DIM)).astype(np.float32)
    ids = np.arange(n, dtype=np.uint64)
    col = DiskCollection(
        str(path),
        dim=_VAMANA_DIM,
        metric=MetricType.L2,
        index_type="disk_vamana",
        vamana_R=32,
        vamana_L=64,
        vamana_alpha=1.2,
        vamana_seed=1234,
        vamana_num_threads=1,
    )
    col.add(vecs, ids)
    col.flush()
    return col


def _build_laser(tmp_path: Path) -> tuple[DiskCollection, np.ndarray]:
    """Build a single-segment Laser collection and return ``(col, vectors)``.

    The vectors come back so tests can use real (in-fixture) queries — Laser
    expects libaio-compatible inputs that match the imported QG.
    """
    src = tmp_path / "src"
    # pylint: disable=import-outside-toplevel
    from fixtures.laser.builder import build_small_laser_artifacts

    _, vectors, labels = build_small_laser_artifacts(
        src,
        n=_LASER_FIXTURE_N,
        dim=_LASER_FIXTURE_DIM,
        R=_LASER_FIXTURE_R,
        seed=1234,
    )
    col = DiskCollection(
        str(tmp_path / "coll"),
        dim=_LASER_FIXTURE_DIM,
        metric=MetricType.L2,
        index_type="disk_laser",
    )
    col.import_laser_segment(str(src), labels)
    return col, vectors


def _engine_params():
    """pytest parametrize ids for the three engines (Laser conditionally skipped)."""
    return [
        pytest.param("disk_flat", id="disk_flat"),
        pytest.param("disk_vamana", id="disk_vamana"),
        pytest.param(
            "disk_laser",
            id="disk_laser",
            marks=pytest.mark.skipif(
                not DISK_LASER_SUPPORTED,
                reason="disk_laser not supported on this build",
            ),
        ),
    ]


def _build_engine_with_query(engine: str, tmp_path: Path) -> tuple[DiskCollection, np.ndarray]:
    """Returns (col, query_2d_shape_(1,dim)_float32)."""
    if engine == "disk_flat":
        col = _build_flat(tmp_path)
        rng = np.random.default_rng(99)
        return col, rng.standard_normal((1, col.dim())).astype(np.float32)
    if engine == "disk_vamana":
        col = _build_vamana(tmp_path)
        rng = np.random.default_rng(99)
        return col, rng.standard_normal((1, col.dim())).astype(np.float32)
    if engine == "disk_laser":
        col, vectors = _build_laser(tmp_path)
        return col, vectors[:1].copy()
    raise ValueError(f"unknown engine: {engine}")


# ---------------------------------------------------------------------------
# Validation tests (engine-agnostic, run on disk_flat)
# ---------------------------------------------------------------------------


def test_wrong_dtype_raises(tmp_path):
    col = _build_flat(tmp_path)
    queries = np.zeros((1, col.dim()), dtype=np.float64)
    with pytest.raises(TypeError, match="float32"):
        col.batch_search(queries, k=5)


def test_1d_queries_raises(tmp_path):
    col = _build_flat(tmp_path)
    queries = np.zeros(col.dim(), dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        col.batch_search(queries, k=5)
    msg = str(exc_info.value)
    assert "ndim" in msg or "2D" in msg


def test_wrong_dim_raises(tmp_path):
    col = _build_flat(tmp_path)
    bogus_dim = col.dim() + 1
    queries = np.zeros((4, bogus_dim), dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        col.batch_search(queries, k=5)
    msg = str(exc_info.value)
    assert str(col.dim()) in msg
    assert str(bogus_dim) in msg


def test_non_contiguous_raises(tmp_path):
    col = _build_flat(tmp_path)
    base = np.zeros((4, col.dim() * 2), dtype=np.float32)
    queries = base[:, ::2]
    assert not queries.flags.c_contiguous
    with pytest.raises((TypeError, ValueError), match="contiguous"):
        col.batch_search(queries, k=5)


def test_k_zero_raises(tmp_path):
    col = _build_flat(tmp_path)
    queries = np.zeros((1, col.dim()), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\bk\b"):
        col.batch_search(queries, k=0)


def test_ef_zero_raises(tmp_path):
    col = _build_flat(tmp_path)
    queries = np.zeros((1, col.dim()), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\bef\b"):
        col.batch_search(queries, k=5, ef=0)


def test_beam_width_zero_raises(tmp_path):
    col = _build_flat(tmp_path)
    queries = np.zeros((1, col.dim()), dtype=np.float32)
    with pytest.raises(ValueError, match="beam_width"):
        col.batch_search(queries, k=5, beam_width=0)


def test_num_threads_negative_raises(tmp_path):
    col = _build_flat(tmp_path)
    queries = np.zeros((1, col.dim()), dtype=np.float32)
    with pytest.raises(ValueError, match="num_threads"):
        col.batch_search(queries, k=5, num_threads=-1)


@pytest.mark.parametrize(
    ("bad_value", "token"),
    [
        (np.float32("nan"), "nan"),
        (np.float32("inf"), "inf"),
        (np.float32("-inf"), "-inf"),
    ],
)
def test_finite_check_raises_with_row_col(tmp_path, bad_value, token):
    col = _build_flat(tmp_path)
    queries = np.zeros((4, col.dim()), dtype=np.float32)
    queries[2, 7] = bad_value
    with pytest.raises(ValueError) as exc_info:
        col.batch_search(queries, k=5)
    msg = str(exc_info.value)
    assert "2" in msg, msg
    assert "7" in msg, msg
    assert token in msg, msg


def test_finite_check_also_applies_to_with_distance(tmp_path):
    col = _build_flat(tmp_path)
    queries = np.zeros((1, col.dim()), dtype=np.float32)
    queries[0, 0] = np.float32("nan")
    with pytest.raises(ValueError, match="nan"):
        col.batch_search_with_distance(queries, k=5)


# ---------------------------------------------------------------------------
# Output shape / dtype / padding contract
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype(tmp_path):
    col = _build_flat(tmp_path)
    queries = np.zeros((4, col.dim()), dtype=np.float32)

    labels = col.batch_search(queries, k=10)
    assert labels.shape == (4, 10)
    assert labels.dtype == np.uint64

    labels2, distances = col.batch_search_with_distance(queries, k=10)
    assert labels2.shape == (4, 10)
    assert labels2.dtype == np.uint64
    assert distances.shape == (4, 10)
    assert distances.dtype == np.float32


def test_padding_sentinels(tmp_path):
    """A 3-vector collection with top_k=10 leaves slots [3, 10) at sentinels."""
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((3, _FLAT_DIM)).astype(np.float32)
    ids = np.array([100, 101, 102], dtype=np.uint64)

    col = DiskCollection(
        str(tmp_path / "flat3"),
        dim=_FLAT_DIM,
        metric=MetricType.L2,
        index_type="disk_flat",
    )
    col.add(vecs, ids)
    col.flush()

    queries = rng.standard_normal((2, _FLAT_DIM)).astype(np.float32)
    labels, distances = col.batch_search_with_distance(queries, k=10)

    for row in range(2):
        seen = sorted(int(x) for x in labels[row, :3])
        assert seen == [100, 101, 102]
        for x in distances[row, :3]:
            assert not math.isnan(float(x))
        for j in range(3, 10):
            assert labels[row, j] == _UINT64_MAX
            assert math.isnan(float(distances[row, j]))


# ---------------------------------------------------------------------------
# Per-engine equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", _engine_params())
def test_single_query_equivalence_labels(engine, tmp_path):
    col, query_2d = _build_engine_with_query(engine, tmp_path)
    serial = col.search(query_2d[0], k=10, ef=100, beam_width=4)
    labels = col.batch_search(query_2d, k=10, ef=100, beam_width=4, num_threads=1)
    for j, hit in enumerate(serial):
        assert labels[0, j] == hit[0], f"engine {engine} col {j}"
    for j in range(len(serial), 10):
        assert labels[0, j] == _UINT64_MAX


@pytest.mark.parametrize("engine", _engine_params())
def test_single_query_equivalence_distances(engine, tmp_path):
    col, query_2d = _build_engine_with_query(engine, tmp_path)
    serial = col.search(query_2d[0], k=10, ef=100, beam_width=4)
    _, distances = col.batch_search_with_distance(query_2d, k=10, ef=100, beam_width=4, num_threads=1)
    if engine == "disk_laser":
        for j in range(len(serial)):
            assert math.isnan(float(distances[0, j])), f"laser col {j}"
    else:
        for j, hit in enumerate(serial):
            dist = hit[1]
            assert np.float32(dist).tobytes() == np.float32(distances[0, j]).tobytes(), (
                f"engine {engine} col {j} distance mismatch: {dist} vs {distances[0, j]}"
            )
    for j in range(len(serial), 10):
        assert math.isnan(float(distances[0, j]))


# ---------------------------------------------------------------------------
# Empty cases
# ---------------------------------------------------------------------------


def test_empty_queries_returns_empty_array(tmp_path):
    col = _build_flat(tmp_path)
    queries = np.empty((0, col.dim()), dtype=np.float32)

    labels = col.batch_search(queries, k=10)
    assert labels.shape == (0, 10)
    assert labels.dtype == np.uint64

    labels2, distances = col.batch_search_with_distance(queries, k=10)
    assert labels2.shape == (0, 10)
    assert distances.shape == (0, 10)
    assert distances.dtype == np.float32


def test_empty_collection_full_sentinels(tmp_path):
    col = DiskCollection(
        str(tmp_path / "empty_flat"),
        dim=_FLAT_DIM,
        metric=MetricType.L2,
        index_type="disk_flat",
    )
    queries = np.zeros((4, _FLAT_DIM), dtype=np.float32)

    labels = col.batch_search(queries, k=10)
    assert (labels == _UINT64_MAX).all()

    labels2, distances = col.batch_search_with_distance(queries, k=10)
    assert (labels2 == _UINT64_MAX).all()
    assert np.isnan(distances).all()


# ---------------------------------------------------------------------------
# num_threads = 0 resolution
# ---------------------------------------------------------------------------


def _serial_label_matrix(col: DiskCollection, queries: np.ndarray, k: int, ef: int) -> np.ndarray:
    out = np.full((queries.shape[0], k), _UINT64_MAX, dtype=np.uint64)
    for i in range(queries.shape[0]):
        for j, (label, _) in enumerate(col.search(queries[i], k=k, ef=ef, beam_width=4)):
            out[i, j] = label
    return out


def test_num_threads_zero_resolves_via_omp(tmp_path, monkeypatch):
    """OMP_NUM_THREADS=4 + num_threads=0 → call succeeds with serial-equivalent result.

    The spec scenario notes that direct verification of "exactly 4 threads"
    would require a test-only telemetry hook (which we deliberately do not
    add); per task 6.2 the thread count itself is hard to verify directly,
    so we check the public contract: no raise + result equals the serial
    baseline.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    col = _build_flat(tmp_path)
    queries = np.random.default_rng(7).standard_normal((4, col.dim())).astype(np.float32)
    expected = _serial_label_matrix(col, queries, k=5, ef=20)
    labels = col.batch_search(queries, k=5, ef=20, beam_width=4, num_threads=0)
    assert np.array_equal(labels, expected)


def test_num_threads_zero_falls_back_to_hardware(tmp_path, monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    col = _build_flat(tmp_path)
    queries = np.random.default_rng(8).standard_normal((4, col.dim())).astype(np.float32)
    expected = _serial_label_matrix(col, queries, k=5, ef=20)
    labels = col.batch_search(queries, k=5, ef=20, beam_width=4, num_threads=0)
    assert np.array_equal(labels, expected)


# ---------------------------------------------------------------------------
# Docstring contract
# ---------------------------------------------------------------------------


def test_docstring_documents_laser_non_scaling():
    doc = DiskCollection.batch_search.__doc__
    assert doc is not None
    assert "disk_laser" in doc
    assert "does not scale" in doc.lower()
    doc2 = DiskCollection.batch_search_with_distance.__doc__
    assert doc2 is not None
    assert "disk_laser" in doc2
    assert "does not scale" in doc2.lower()


def test_docstring_documents_omp_resolution():
    doc = DiskCollection.batch_search.__doc__
    assert "OMP_NUM_THREADS" in doc
    doc2 = DiskCollection.batch_search_with_distance.__doc__
    # batch_search_with_distance MAY satisfy by either restating or by
    # cross-referencing batch_search; the spec test phrases this as "documents
    # the num_threads = 0 auto resolution via OMP_NUM_THREADS".
    assert "OMP_NUM_THREADS" in doc2 or "batch_search" in doc2


# ---------------------------------------------------------------------------
# Laser 8-thread baseline (concurrent test, Laser-only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DISK_LASER_SUPPORTED, reason="disk_laser not supported on this build")
def test_disk_laser_batch_8threads_matches_serial_baseline(tmp_path):
    """8-thread batch_search agrees with serial baseline on a Laser collection.

    Laser is mutex-serialized internally so this test verifies correctness
    (labels match) — not throughput scaling, which is intentionally not
    delivered for Laser per spec D3. The query batch matches the spec
    scenario (N=64) and the call must complete inside the 60-second
    wall-clock budget the spec mandates; we measure the budget inline
    rather than introducing a pytest-timeout dependency.
    """
    col, vectors = _build_laser(tmp_path)
    n = 64
    queries = vectors[:n].copy()
    expected = np.full((n, 10), _UINT64_MAX, dtype=np.uint64)
    for i in range(n):
        for j, (label, _) in enumerate(col.search(queries[i], k=10, ef=100, beam_width=4)):
            expected[i, j] = label
    started = time.monotonic()
    labels = col.batch_search(queries, k=10, ef=100, beam_width=4, num_threads=8)
    elapsed = time.monotonic() - started
    assert elapsed < 60.0, f"batch_search did not complete within 60s budget (took {elapsed:.2f}s)"
    assert np.array_equal(labels, expected)


# ---------------------------------------------------------------------------
# Sanity: in-memory Index batch_search untouched
# ---------------------------------------------------------------------------


def test_in_memory_index_batch_search_still_present():
    """alayalite.Index keeps its own batch_search; this change does not alias or remove it."""
    # pylint: disable=import-outside-toplevel
    from alayalite import Index

    assert hasattr(Index, "batch_search")
    assert hasattr(Index, "batch_search_with_distance")
