# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for alayalite.bench._datasets within the bench test suite."""

import struct
from pathlib import Path

import numpy as np
import pytest
from alayalite.bench._datasets import (
    DatasetMissing,
    DatasetSpec,
    _distance_order,
    _load_fbin,
    _load_fvecs,
    _load_ibin,
    _load_ivecs,
    _sha16_arrays,
    _sha16_directory,
    _sha16_files,
    exact_ground_truth,
    load_gist1m,
    load_laser_files,
    load_sift1m,
    load_synth,
)

# ── binary format helpers ─────────────────────────────────────────────────────


def _write_vecs(path: Path, vecs: np.ndarray, dtype) -> None:
    """Write *vecs* format (per-row int32 dim prefix): fvecs (float32) or ivecs (int32)."""
    vecs = np.ascontiguousarray(vecs, dtype=dtype)
    with path.open("wb") as f:
        for row in vecs:
            f.write(np.int32(row.shape[0]).tobytes())
            f.write(row.tobytes())


def _write_bin(path: Path, vecs: np.ndarray, dtype) -> None:
    """Write *bin* format (single ``<II`` header + body): fbin (float32) or ibin (uint32)."""
    vecs = np.ascontiguousarray(vecs, dtype=dtype)
    n, dim = vecs.shape
    with path.open("wb") as f:
        f.write(struct.pack("<II", n, dim))
        f.write(vecs.tobytes())


# ── _sha16_arrays ─────────────────────────────────────────────────────────────


def test_sha16_arrays_returns_16_hex_chars():
    a = np.zeros((3, 2), dtype=np.float32)
    result = _sha16_arrays(a)
    assert len(result) == 16
    assert all(c in "0123456789abcdef" for c in result)


def test_sha16_arrays_bytes_input():
    assert len(_sha16_arrays(b"hello", b"world")) == 16


def test_sha16_arrays_deterministic():
    a = np.ones((4,), dtype=np.float32)
    assert _sha16_arrays(a) == _sha16_arrays(a)


# ── _sha16_files ──────────────────────────────────────────────────────────────


def test_sha16_files_hashes_content(tmp_path):
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"hello")
    f2.write_bytes(b"world")
    result = _sha16_files([f1, f2])
    assert len(result) == 16
    f1.write_bytes(b"HELLO")
    assert _sha16_files([f1, f2]) != result


# ── _sha16_directory ──────────────────────────────────────────────────────────


def test_sha16_directory_skips_subdirs(tmp_path):
    # Regression: subdirs must not be hashed (only regular files).
    (tmp_path / "sub").mkdir()
    (tmp_path / "file.bin").write_bytes(b"data")
    result = _sha16_directory(tmp_path)
    assert len(result) == 16


# ── _distance_order ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "metric, vecs, q",
    [
        ("COS", [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], [1.0, 0.0]),
        ("IP", [[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [1.0, 0.0]),
        ("L2", [[0.0, 0.0], [10.0, 10.0], [1.0, 0.0]], [0.0, 0.0]),
    ],
)
def test_distance_order_picks_first_index(metric, vecs, q):
    order = _distance_order(np.asarray(vecs, dtype=np.float32), np.asarray(q, dtype=np.float32), metric, 2)
    assert order[0] == 0


# ── exact_ground_truth ────────────────────────────────────────────────────────


def test_exact_ground_truth_empty_raises():
    with pytest.raises(ValueError, match="empty dataset"):
        exact_ground_truth(
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.uint64),
            np.ones((2, 4), dtype=np.float32),
            metric="L2",
        )


def test_exact_ground_truth_basic():
    vecs = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    labels = np.arange(3, dtype=np.uint64)
    queries = np.array([[0.1, 0.0]], dtype=np.float32)
    gt = exact_ground_truth(vecs, labels, queries, metric="L2", k=2)
    assert gt.shape == (1, 2)
    assert gt[0, 0] == 0


# ── load_synth ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "n, dim, queries, match",
    [
        (0, 4, 5, "--n must be > 0"),
        (10, 0, 5, "--dim must be > 0"),
        (10, 4, 0, "--queries must be > 0"),
    ],
)
def test_load_synth_zero_arg_raises(n, dim, queries, match):
    with pytest.raises(ValueError, match=match):
        load_synth(n, dim, queries, 42)


def test_load_synth_ok():
    ds = load_synth(20, 4, 5, 42)
    assert isinstance(ds, DatasetSpec)
    assert ds.vectors.shape == (20, 4)
    assert ds.n == 20
    assert ds.dim == 4


# ── _load_fvecs / _load_ivecs ─────────────────────────────────────────────────


def test_load_fvecs_success(tmp_path):
    vecs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    p = tmp_path / "test.fvecs"
    _write_vecs(p, vecs, np.float32)
    np.testing.assert_array_almost_equal(_load_fvecs(p), vecs)


def test_load_fvecs_empty_raises(tmp_path):
    p = tmp_path / "empty.fvecs"
    p.write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty fvecs"):
        _load_fvecs(p)


def test_load_fvecs_malformed_raises(tmp_path):
    p = tmp_path / "bad.fvecs"
    # dim=3 but only 2 floats follow → size % row_width != 0
    p.write_bytes(np.int32(3).tobytes() + np.float32(1.0).tobytes() + np.float32(2.0).tobytes())
    with pytest.raises(RuntimeError, match="malformed fvecs"):
        _load_fvecs(p)


def test_load_ivecs_success(tmp_path):
    vecs = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    p = tmp_path / "test.ivecs"
    _write_vecs(p, vecs, np.int32)
    loaded = _load_ivecs(p)
    assert loaded.dtype == np.uint64
    np.testing.assert_array_equal(loaded, vecs.astype(np.uint64))


def test_load_ivecs_empty_raises(tmp_path):
    p = tmp_path / "empty.ivecs"
    p.write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty ivecs"):
        _load_ivecs(p)


def test_load_ivecs_malformed_raises(tmp_path):
    p = tmp_path / "bad.ivecs"
    # dim=3 but only 1 int32 follows
    p.write_bytes(np.int32(3).tobytes() + np.int32(0).tobytes())
    with pytest.raises(RuntimeError, match="malformed ivecs"):
        _load_ivecs(p)


# ── _load_fbin / _load_ibin ───────────────────────────────────────────────────


def test_load_fbin_success(tmp_path):
    vecs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    p = tmp_path / "test.fbin"
    _write_bin(p, vecs, np.float32)
    np.testing.assert_array_almost_equal(_load_fbin(p), vecs)


def test_load_fbin_truncated_header_raises(tmp_path):
    p = tmp_path / "bad_hdr.fbin"
    p.write_bytes(b"\x00\x01")  # only 2 bytes; need 8
    with pytest.raises(RuntimeError, match="truncated fbin header"):
        _load_fbin(p)


def test_load_fbin_truncated_body_raises(tmp_path):
    p = tmp_path / "bad_body.fbin"
    with p.open("wb") as f:
        f.write(struct.pack("<II", 10, 4))  # n=10, dim=4 but no body
    with pytest.raises(RuntimeError, match="truncated fbin body"):
        _load_fbin(p)


def test_load_ibin_success(tmp_path):
    vecs = np.array([[0, 1], [2, 3]], dtype=np.uint32)
    p = tmp_path / "test.ibin"
    _write_bin(p, vecs, np.uint32)
    loaded = _load_ibin(p)
    assert loaded.dtype == np.uint64
    np.testing.assert_array_equal(loaded, vecs.astype(np.uint64))


def test_load_ibin_truncated_header_raises(tmp_path):
    p = tmp_path / "bad_hdr.ibin"
    p.write_bytes(b"\x01\x02")  # only 2 bytes
    with pytest.raises(RuntimeError, match="truncated ibin header"):
        _load_ibin(p)


def test_load_ibin_truncated_body_raises(tmp_path):
    p = tmp_path / "bad_body.ibin"
    with p.open("wb") as f:
        f.write(struct.pack("<II", 5, 5))  # n=5, dim=5 but no body
    with pytest.raises(RuntimeError, match="truncated ibin body"):
        _load_ibin(p)


# ── load_sift1m / load_gist1m ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "loader, prefix",
    [(load_sift1m, "sift1m"), (load_gist1m, "gist1m")],
)
def test_named_dataset_missing_raises(tmp_path, loader, prefix):
    with pytest.raises(DatasetMissing, match=prefix):
        loader(tmp_path / "nonexistent")


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "loader": load_sift1m,
                "base_name": "sift_base.fvecs",
                "query_name": "sift_query.fvecs",
                "gt_name": "sift_groundtruth.ivecs",
                "vec_dtype": np.float32,
                "gt_dtype": np.int32,
                "vec_writer": _write_vecs,
                "gt_writer": _write_vecs,
                "gt_value": np.arange(30, dtype=np.int32).reshape(3, 10),
            },
            id="sift1m",
        ),
        pytest.param(
            {
                "loader": load_gist1m,
                "base_name": "gist_base.fbin",
                "query_name": "gist_query.fbin",
                "gt_name": "gist_gt.ibin",
                "vec_dtype": np.float32,
                "gt_dtype": np.uint32,
                "vec_writer": _write_bin,
                "gt_writer": _write_bin,
                "gt_value": np.arange(16, dtype=np.uint32).reshape(2, 8),
            },
            id="gist1m",
        ),
    ],
)
def test_named_dataset_success(tmp_path, case):
    rng = np.random.default_rng(0)
    loader = case["loader"]
    gt_value = case["gt_value"]
    n, dim = gt_value.shape[1], 4
    vecs = rng.standard_normal((n, dim)).astype(case["vec_dtype"])
    queries = rng.standard_normal((gt_value.shape[0], dim)).astype(case["vec_dtype"])
    case["vec_writer"](tmp_path / case["base_name"], vecs, case["vec_dtype"])
    case["vec_writer"](tmp_path / case["query_name"], queries, case["vec_dtype"])
    case["gt_writer"](tmp_path / case["gt_name"], gt_value, case["gt_dtype"])
    ds = loader(tmp_path)
    assert ds.vectors.shape == (n, dim)
    assert ds.queries.shape == (gt_value.shape[0], dim)
    assert ds.ground_truth is not None


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "loader": load_sift1m,
                "base_name": "sift_base.fvecs",
                "query_name": "sift_query.fvecs",
                "vec_dtype": np.float32,
                "writer": _write_vecs,
            },
            id="sift1m",
        ),
        pytest.param(
            {
                "loader": load_gist1m,
                "base_name": "gist_base.fbin",
                "query_name": "gist_query.fbin",
                "vec_dtype": np.float32,
                "writer": _write_bin,
            },
            id="gist1m",
        ),
    ],
)
def test_named_dataset_no_gt(tmp_path, case):
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((5, 4)).astype(case["vec_dtype"])
    queries = rng.standard_normal((2, 4)).astype(case["vec_dtype"])
    case["writer"](tmp_path / case["base_name"], vecs, case["vec_dtype"])
    case["writer"](tmp_path / case["query_name"], queries, case["vec_dtype"])
    assert case["loader"](tmp_path).ground_truth is None


# ── load_laser_files ──────────────────────────────────────────────────────────


def test_load_laser_files_from_vectors_path(tmp_path):
    rng = np.random.default_rng(4)
    vecs = rng.standard_normal((12, 4)).astype(np.float32)
    vp = tmp_path / "vecs.fbin"
    _write_bin(vp, vecs, np.float32)
    ds = load_laser_files(n=0, dim=4, query_count=3, seed=42, vectors_path=vp)
    assert ds.vectors.shape[1] == 4
    assert ds.n == 12


def test_load_laser_files_queries_dim_mismatch(tmp_path):
    rng = np.random.default_rng(5)
    vecs = rng.standard_normal((10, 4)).astype(np.float32)
    queries = rng.standard_normal((3, 8)).astype(np.float32)  # wrong dim
    vp = tmp_path / "v.fbin"
    qp = tmp_path / "q.fbin"
    _write_bin(vp, vecs, np.float32)
    _write_bin(qp, queries, np.float32)
    with pytest.raises(ValueError, match="dim.*does not match"):
        load_laser_files(n=0, dim=4, query_count=3, seed=42, vectors_path=vp, queries_path=qp)


def test_load_laser_files_ground_truth_path(tmp_path):
    gt = np.arange(6, dtype=np.uint32).reshape(2, 3)
    gp = tmp_path / "gt.ibin"
    _write_bin(gp, gt, np.uint32)
    ds = load_laser_files(n=5, dim=4, query_count=2, seed=42, ground_truth_path=gp)
    assert ds.ground_truth is not None
    assert ds.ground_truth.shape == (2, 3)


def test_load_laser_files_no_vectors_no_queries():
    ds = load_laser_files(n=5, dim=4, query_count=3, seed=42)
    assert ds.vectors.shape == (0, 4)
    assert ds.queries.shape == (3, 4)
    assert ds.ground_truth is None


def test_load_laser_files_laser_src_dir(tmp_path):
    src_dir = tmp_path / "laser_src"
    src_dir.mkdir()
    (src_dir / "shard.bin").write_bytes(b"fake shard data")
    ds = load_laser_files(n=5, dim=4, query_count=2, seed=42, laser_src_dir=src_dir)
    assert len(ds.sha16) == 16
