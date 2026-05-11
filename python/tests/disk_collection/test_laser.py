# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the alayalite.DiskCollection(index_type="disk_laser") binding.

The fixture builder is lazily imported inside `_build_fixture` so the
module-level `import` is safe on unsupported builds (Linux+OFF / macOS /
Windows). Per design D6, every supported-path test is gated on
`DISK_LASER_SUPPORTED` runtime probe; a single negative-path test
(`test_disk_laser_unsupported_platform`) runs only on unsupported
builds and pins the dual-substring rejection contract.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from _laser_support import DISK_LASER_SUPPORTED  # noqa: E402
from alayalite import DiskCollection, MetricType

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="DiskCollection v1 is POSIX-only; disk_laser additionally requires Linux+libaio",
)

_SEG_PREFIX = "seg_00000001"
_FIXTURE_DIM = 128
_FIXTURE_R = 64


def _build_fixture(target_dir: Path, *, n: int = 256, seed: int = 1234):
    """Lazily import the builder so unsupported builds skip cleanly."""
    # pylint: disable=import-outside-toplevel
    # The builder transitively imports alayalite.laser / vamana, which are
    # not loadable on unsupported builds. Keeping the import here lets the
    # test module load on those builds so the unsupported-path test
    # (test_disk_laser_unsupported_platform) can still run.
    from fixtures.laser.builder import build_small_laser_artifacts

    return build_small_laser_artifacts(
        target_dir,
        seg_basename=_SEG_PREFIX,
        n=n,
        dim=_FIXTURE_DIM,
        R=_FIXTURE_R,
        seed=seed,
    )


def _make_disk_laser(tmp_path: Path, name: str = "coll") -> DiskCollection:
    return DiskCollection(
        path=str(tmp_path / name),
        dim=_FIXTURE_DIM,
        metric=MetricType.L2,
        index_type="disk_laser",
    )


# ---------------------------------------------------------------------------
# Tests gated on supported builds (Linux + ALAYA_ENABLE_LASER=ON).
# ---------------------------------------------------------------------------

requires_laser = pytest.mark.skipif(
    not DISK_LASER_SUPPORTED,
    reason="disk_laser is not supported on this build/platform",
)


@requires_laser
def test_disk_laser_import_basic(tmp_path):
    """Import a fresh segment, search, assert NaN distances + valid labels."""
    src_dir = tmp_path / "src"
    _, _, labels = _build_fixture(src_dir)
    col = _make_disk_laser(tmp_path)
    col.import_laser_segment(str(src_dir), labels)
    assert col.size() == labels.shape[0]

    rng = np.random.default_rng(42)
    query = rng.standard_normal(_FIXTURE_DIM).astype(np.float32)
    hits = col.search(query, k=10, ef=128, beam_width=4)
    assert isinstance(hits, list)
    assert 0 < len(hits) <= 10
    label_set = set(labels.tolist())
    for label, distance in hits:
        assert label in label_set, f"returned label {label} not in imported labels"
        # spec: disk_laser distances are NaN (LASER's adapter does not surface
        # distances through the v1 surface).
        assert math.isnan(distance), f"disk_laser distance must be NaN (got {distance})"


@requires_laser
def test_disk_laser_reopen(tmp_path):
    """Reopening a disk_laser collection returns identical search results."""
    src_dir = tmp_path / "src"
    _, _, labels = _build_fixture(src_dir)
    path = tmp_path / "coll"
    col = DiskCollection(
        path=str(path),
        dim=_FIXTURE_DIM,
        metric=MetricType.L2,
        index_type="disk_laser",
    )
    col.import_laser_segment(str(src_dir), labels)

    rng = np.random.default_rng(7)
    query = rng.standard_normal(_FIXTURE_DIM).astype(np.float32)
    hits1 = col.search(query, k=5, ef=128, beam_width=4)
    del col

    col2 = DiskCollection.open(str(path))
    assert col2.size() == labels.shape[0]
    assert col2.dim() == _FIXTURE_DIM
    hits2 = col2.search(query, k=5, ef=128, beam_width=4)
    # Compare labels (distances are NaN so direct tuple equality fails).
    assert [label for label, _ in hits1] == [label for label, _ in hits2]


@requires_laser
def test_disk_laser_external_labels(tmp_path):
    """Search returns external labels (not 0..n-1 internal PIDs)."""
    src_dir = tmp_path / "src"
    _, _, fixture_labels = _build_fixture(src_dir, n=256)
    # Replace labels with a sparse set so we can trivially detect PID leakage.
    sparse_labels = np.array(
        [7, 13, 21, 42, 99, 1234] + (256 - 6) * [0],
        dtype=np.uint64,
    )
    sparse_labels[6:] = np.arange(2_000_000, 2_000_000 + (256 - 6), dtype=np.uint64)
    assert sparse_labels.shape == fixture_labels.shape

    col = _make_disk_laser(tmp_path)
    col.import_laser_segment(str(src_dir), sparse_labels)

    rng = np.random.default_rng(2)
    query = rng.standard_normal(_FIXTURE_DIM).astype(np.float32)
    hits = col.search(query, k=10, ef=128)
    label_set = set(sparse_labels.tolist())
    for label, _ in hits:
        assert label in label_set, f"returned label {label} not in imported sparse label set"


@requires_laser
def test_disk_laser_add_flush_unsupported(tmp_path):
    """add() and flush() reject disk_laser collections at the binding boundary."""
    col = _make_disk_laser(tmp_path)
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((4, _FIXTURE_DIM)).astype(np.float32)
    ids = np.array([10, 20, 30, 40], dtype=np.uint64)
    with pytest.raises(RuntimeError) as exc_info:
        col.add(vectors, ids)
    msg = str(exc_info.value)
    assert "disk_laser" in msg
    assert "not implemented in v1" in msg
    assert "import_laser_segment" in msg

    with pytest.raises(RuntimeError) as exc_info:
        col.flush()
    msg = str(exc_info.value)
    assert "disk_laser" in msg
    assert "not implemented in v1" in msg
    assert "import_laser_segment" in msg


@requires_laser
def test_disk_laser_duplicate_label_within_batch(tmp_path):
    src_dir = tmp_path / "src"
    _, _, labels = _build_fixture(src_dir)
    bad_labels = labels.copy()
    bad_labels[5] = bad_labels[0]  # duplicate
    col = _make_disk_laser(tmp_path)
    with pytest.raises(ValueError):
        col.import_laser_segment(str(src_dir), bad_labels)
    # No segment SHALL be created on disk.
    seg_dirs = list((tmp_path / "coll" / "segments").glob("seg_*"))
    assert not seg_dirs


@requires_laser
def test_disk_laser_duplicate_label_cross_segment(tmp_path):
    src1 = tmp_path / "src1"
    src2 = tmp_path / "src2"
    _, _, labels1 = _build_fixture(src1)
    _, _, labels2 = _build_fixture(src2, seed=4321)
    # Reuse one label from labels1 in labels2 to trigger the cross-segment
    # uniqueness check.
    labels2[0] = labels1[10]
    col = _make_disk_laser(tmp_path)
    col.import_laser_segment(str(src1), labels1)
    with pytest.raises(ValueError):
        col.import_laser_segment(str(src2), labels2)
    seg_dirs = sorted((tmp_path / "coll" / "segments").glob("seg_*"))
    assert len(seg_dirs) == 1, f"expected only seg_00000001 on disk, got {seg_dirs}"


@requires_laser
@pytest.mark.parametrize(
    "labels_factory, exc_type, expected_substring",
    [
        (lambda labels: labels.astype(np.int32), TypeError, "uint64"),
        (lambda labels: np.empty(2 * labels.shape[0], dtype=np.uint64)[::2], TypeError, "contiguous"),
        (lambda labels: labels.reshape(-1, 1), ValueError, ""),
        (lambda labels: np.empty((0,), dtype=np.uint64), ValueError, ""),
    ],
    ids=["int32", "non_contiguous", "two_d", "empty"],
)
def test_disk_laser_invalid_labels_throw(tmp_path, labels_factory, exc_type, expected_substring):
    src_dir = tmp_path / "src"
    _, _, labels = _build_fixture(src_dir)
    bad = labels_factory(labels)
    col = _make_disk_laser(tmp_path)
    with pytest.raises(exc_type) as exc_info:
        col.import_laser_segment(str(src_dir), bad)
    if expected_substring:
        assert expected_substring in str(exc_info.value).lower()


@requires_laser
@pytest.mark.parametrize(
    "make_path",
    [
        lambda tmp: tmp / "absolutely_not_there",
        lambda tmp: tmp / "regular_file.txt",
        lambda tmp: tmp / "symlink_to_file",
    ],
    ids=["nonexistent", "regular_file", "symlink_to_file"],
)
def test_disk_laser_invalid_src_dir_throws(tmp_path, make_path):
    # Set up a real src_dir so we can instantiate the builder once for all
    # parametrize cases, and a sibling regular_file/symlink target for the
    # invalid-path scenarios.
    real_src = tmp_path / "real_src"
    _, _, labels = _build_fixture(real_src)
    col = _make_disk_laser(tmp_path)

    bad_path = make_path(tmp_path)
    if "regular_file" in str(bad_path):
        bad_path.write_text("regular file, not a directory", encoding="utf-8")
    elif "symlink" in str(bad_path):
        # symlink target is the regular file (non-directory) we just wrote
        target = tmp_path / "regular_file.txt"
        if not target.exists():
            target.write_text("regular file, not a directory", encoding="utf-8")
        bad_path.symlink_to(target)

    with pytest.raises(ValueError) as exc_info:
        col.import_laser_segment(str(bad_path), labels)
    assert "src_dir" in str(exc_info.value)
    # No segment SHALL be created on disk.
    seg_dirs = list((tmp_path / "coll" / "segments").glob("seg_*"))
    assert not seg_dirs


@requires_laser
def test_disk_laser_query_validation(tmp_path):
    src_dir = tmp_path / "src"
    _, _, labels = _build_fixture(src_dir)
    col = _make_disk_laser(tmp_path)
    col.import_laser_segment(str(src_dir), labels)

    base_query = np.zeros(_FIXTURE_DIM, dtype=np.float32)

    # (a) wrong dtype.
    with pytest.raises(TypeError):
        col.search(base_query.astype(np.float64), k=10, ef=64)

    # (b) non-contiguous.
    big = np.zeros(2 * _FIXTURE_DIM, dtype=np.float32)
    bad_q = big[::2]
    assert not bad_q.flags["C_CONTIGUOUS"]
    with pytest.raises((TypeError, ValueError)) as exc_info:
        col.search(bad_q, k=10, ef=64)
    assert "contiguous" in str(exc_info.value).lower()

    # (c) wrong shape.
    with pytest.raises(ValueError):
        col.search(np.zeros(_FIXTURE_DIM + 1, dtype=np.float32), k=10, ef=64)

    # (d/e/f) NaN / +Inf / -Inf at specific positions.
    for index, bad_val, name in [(5, np.nan, "nan"), (0, np.inf, "inf"), (10, -np.inf, "-inf")]:
        bad_query = base_query.copy()
        bad_query[index] = bad_val
        with pytest.raises(ValueError) as exc_info:
            col.search(bad_query, k=10, ef=64)
        msg = str(exc_info.value).lower()
        assert f"[{index}]" in msg
        assert name in msg


@requires_laser
def test_disk_laser_search_beam_width_validation(tmp_path):
    src_dir = tmp_path / "src"
    _, _, labels = _build_fixture(src_dir)
    col = _make_disk_laser(tmp_path)
    col.import_laser_segment(str(src_dir), labels)

    query = np.zeros(_FIXTURE_DIM, dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        col.search(query, k=10, ef=64, beam_width=0)
    assert "beam_width" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        col.search(query, k=10, ef=64, beam_width=-1)
    assert "beam_width" in str(exc_info.value)


@requires_laser
def test_disk_laser_copy_false_raises(tmp_path):
    src_dir = tmp_path / "src"
    _, _, labels = _build_fixture(src_dir)
    col = _make_disk_laser(tmp_path)
    with pytest.raises(NotImplementedError) as exc_info:
        col.import_laser_segment(str(src_dir), labels, copy=False)
    msg = str(exc_info.value)
    assert "copy=True" in msg
    assert "import_laser_segment" in msg
    assert "LaserSegmentImportParams" in msg
    seg_dirs = list((tmp_path / "coll" / "segments").glob("seg_*"))
    assert not seg_dirs


# ---------------------------------------------------------------------------
# Cross-engine query-finite check (spec scenarios 6.9b / 6.9c).
# These run on every supported build (disk_flat is always supported,
# disk_vamana is always supported on Linux+libaio).
# ---------------------------------------------------------------------------


def _ids(n, base=1000):
    return np.arange(base, base + n, dtype=np.uint64)


def _rand_vectors(n, dim, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


@pytest.mark.parametrize("bad_value, label", [(np.nan, "nan"), (np.inf, "inf"), (-np.inf, "-inf")])
def test_query_finite_check_applies_to_disk_flat(tmp_path, bad_value, label):
    """Pin the cross-engine uniformity scenario `NaN check applies to disk_flat`."""
    col = DiskCollection(
        path=str(tmp_path / "coll"),
        dim=8,
        metric=MetricType.L2,
        index_type="disk_flat",
    )
    col.add(_rand_vectors(8, 8), _ids(8))
    col.flush()
    bad_query = np.zeros(8, dtype=np.float32)
    bad_query[3] = bad_value
    with pytest.raises(ValueError) as exc_info:
        col.search(bad_query, k=5)
    msg = str(exc_info.value).lower()
    assert "[3]" in msg
    assert label in msg


@pytest.mark.parametrize("bad_value, label", [(np.nan, "nan"), (np.inf, "inf"), (-np.inf, "-inf")])
def test_query_finite_check_applies_to_disk_vamana(tmp_path, bad_value, label):
    """Pin the cross-engine uniformity scenario `NaN check applies to disk_vamana`.

    Existing per-vamana check (in `vamana_segment_searcher.hpp`) keeps firing
    on the C++ surface, but Python callers see the Python-side check first.
    """
    col = DiskCollection(
        path=str(tmp_path / "coll"),
        dim=16,
        metric=MetricType.L2,
        index_type="disk_vamana",
    )
    col.add(_rand_vectors(64, 16), _ids(64))
    col.flush()
    bad_query = np.zeros(16, dtype=np.float32)
    bad_query[7] = bad_value
    with pytest.raises(ValueError) as exc_info:
        col.search(bad_query, k=5, ef=64)
    msg = str(exc_info.value).lower()
    assert "[7]" in msg
    assert label in msg


# ---------------------------------------------------------------------------
# Negative path: only runs on unsupported builds.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    DISK_LASER_SUPPORTED,
    reason="run only on unsupported builds; supported-build positive path is covered above",
)
def test_disk_laser_unsupported_platform(tmp_path):
    """On unsupported builds, both constructor and open() raise the dual-substring error."""
    with pytest.raises(ValueError) as exc_info:
        DiskCollection(
            path=str(tmp_path / "coll"),
            dim=128,
            metric=MetricType.L2,
            index_type="disk_laser",
        )
    msg = str(exc_info.value)
    assert "disk_laser" in msg
    assert "not implemented in v1" in msg
    assert not (tmp_path / "coll").exists()

    # Build a manifest by hand (no need for the Python ctor) and pass it to open().
    manifest_dir = tmp_path / "manifest_only"
    (manifest_dir / "segments").mkdir(parents=True)
    (manifest_dir / "collection_manifest.txt").write_text(
        "\n".join(
            [
                "version=1",
                "dim=128",
                "metric=L2",
                "index_type=disk_laser",
                "next_segment_id=1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc_info:
        DiskCollection.open(str(manifest_dir))
    msg = str(exc_info.value)
    assert "disk_laser" in msg
    assert "not implemented in v1" in msg
