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

"""Tests for the alayalite.DiskCollection Python binding."""

import sys

import numpy as np
import pytest
from _laser_support import DISK_LASER_SUPPORTED
from alayalite import DiskCollection, MetricType

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="DiskCollection v1 is POSIX-only")


def _rand_vectors(n, dim, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _ids(n, base=1000):
    return np.arange(base, base + n, dtype=np.uint64)


def test_disk_collection_basic(tmp_path):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=8, metric=MetricType.L2, index_type="disk_flat")
    v = _rand_vectors(20, 8)
    ids = _ids(20)
    col.add(v, ids)
    col.flush()
    assert col.size() == 20
    assert col.dim() == 8

    q = _rand_vectors(1, 8, seed=7)[0]
    hits = col.search(q, k=5)
    assert isinstance(hits, list)
    assert len(hits) == 5
    for label, dist in hits:
        assert isinstance(label, int)
        assert isinstance(dist, float)
    distances = [d for _, d in hits]
    assert distances == sorted(distances)


def test_disk_collection_reopen(tmp_path):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=8, metric=MetricType.L2, index_type="disk_flat")
    v = _rand_vectors(50, 8)
    ids = _ids(50)
    col.add(v, ids)
    col.flush()
    q = _rand_vectors(1, 8, seed=99)[0]
    hits1 = col.search(q, k=5)
    del col

    col2 = DiskCollection.open(path)
    assert col2.size() == 50
    hits2 = col2.search(q, k=5)
    assert hits1 == hits2


def test_disk_collection_matches_bruteforce(tmp_path):
    path = str(tmp_path / "coll")
    dim = 32
    col = DiskCollection(path=path, dim=dim, metric=MetricType.L2, index_type="disk_flat")
    v = _rand_vectors(1000, dim, seed=1)
    ids = _ids(1000)
    col.add(v, ids)
    col.flush()

    q = _rand_vectors(1, dim, seed=2)[0]
    hits = col.search(q, k=10)
    # Brute-force.
    diffs = v - q
    dists = (diffs * diffs).sum(axis=1)
    order = np.argsort(dists, kind="stable")[:10]
    expected = [(int(ids[i]), float(dists[i])) for i in order]
    for got, want in zip(hits, expected):
        assert got[0] == want[0]
        assert got[1] == pytest.approx(want[1], rel=1e-3, abs=1e-3)


@pytest.mark.parametrize(
    "vectors_dtype, ids_dtype, expected_msg_token",
    [
        (np.float64, np.uint64, "float32"),
        (np.int32, np.uint64, "float32"),
        (np.float32, np.int32, "uint64"),
        (np.float32, np.uint32, "uint64"),
    ],
)
def test_disk_collection_add_dtype_errors(tmp_path, vectors_dtype, ids_dtype, expected_msg_token):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_flat")
    v = np.zeros((3, 4), dtype=vectors_dtype)
    ids = np.array([1, 2, 3], dtype=ids_dtype)
    with pytest.raises((TypeError, ValueError)) as exc_info:
        col.add(v, ids)
    assert expected_msg_token in str(exc_info.value), (
        f"error message must mention {expected_msg_token!r}: {exc_info.value}"
    )


def test_disk_collection_add_non_contiguous_vectors_raises(tmp_path):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_flat")
    big = np.zeros((3, 8), dtype=np.float32)
    v = big[:, :4]  # strided view, not C-contiguous
    ids = np.array([1, 2, 3], dtype=np.uint64)
    assert not v.flags["C_CONTIGUOUS"]
    with pytest.raises((TypeError, ValueError)) as exc_info:
        col.add(v, ids)
    assert "contiguous" in str(exc_info.value).lower()


def test_disk_collection_add_non_contiguous_ids_raises(tmp_path):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_flat")
    v = _rand_vectors(3, 4)
    big_ids = np.zeros(6, dtype=np.uint64)
    ids = big_ids[::2]  # strided view of ids
    assert not ids.flags["C_CONTIGUOUS"]
    with pytest.raises((TypeError, ValueError)) as exc_info:
        col.add(v, ids)
    assert "contiguous" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "vshape, ishape",
    [
        ((3,), (3,)),  # 1D vectors
        ((3, 8), (3,)),  # mismatched dim
        ((3, 4), (4,)),  # mismatched n
    ],
)
def test_disk_collection_add_shape_errors(tmp_path, vshape, ishape):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_flat")
    v = np.zeros(vshape, dtype=np.float32)
    ids = np.zeros(ishape, dtype=np.uint64)
    with pytest.raises(ValueError):
        col.add(v, ids)


def test_disk_collection_search_dtype_errors(tmp_path):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_flat")
    col.add(_rand_vectors(2, 4), _ids(2))
    col.flush()
    bad_q = np.zeros(4, dtype=np.float64)
    with pytest.raises(TypeError):
        col.search(bad_q, k=1)


def test_disk_collection_search_non_contiguous_query_raises(tmp_path):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_flat")
    col.add(_rand_vectors(2, 4), _ids(2))
    col.flush()
    big = np.zeros(8, dtype=np.float32)
    bad_q = big[::2]  # strided
    assert not bad_q.flags["C_CONTIGUOUS"]
    with pytest.raises((TypeError, ValueError)):
        col.search(bad_q, k=1)


@pytest.mark.parametrize(
    "qshape",
    [(2, 4), (8,), (0,)],  # 2D, wrong dim, empty
)
def test_disk_collection_search_shape_errors(tmp_path, qshape):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_flat")
    col.add(_rand_vectors(2, 4), _ids(2))
    col.flush()
    bad_q = np.zeros(qshape, dtype=np.float32)
    with pytest.raises(ValueError):
        col.search(bad_q, k=1)


@pytest.mark.parametrize("metric", [MetricType.L2, MetricType.IP, MetricType.COS])
def test_disk_collection_metric_l2_ip_cos(tmp_path, metric):
    path = str(tmp_path / f"coll_{metric}")
    dim = 8
    col = DiskCollection(path=path, dim=dim, metric=metric, index_type="disk_flat")
    v = _rand_vectors(50, dim, seed=int(metric))
    ids = _ids(50)
    col.add(v, ids)
    col.flush()

    q = _rand_vectors(1, dim, seed=999)[0]
    hits = col.search(q, k=3)
    assert len(hits) == 3
    distances = [d for _, d in hits]
    assert distances == sorted(distances)

    if metric == MetricType.COS:
        # COS distance contract: -⟨q_normalized, v_normalized⟩
        qn = q / np.linalg.norm(q)
        for label, dist in hits:
            row_idx = int(np.argwhere(ids == label)[0, 0])
            vn = v[row_idx] / np.linalg.norm(v[row_idx])
            expected = -float(np.dot(qn, vn))
            assert dist == pytest.approx(expected, abs=1e-4)


def test_disk_collection_cos_distance_docstring():
    doc = DiskCollection.search.__doc__
    assert "L2: squared distance" in doc
    assert "IP: negative inner product" in doc
    assert "COS: negative cosine similarity" in doc
    assert "smaller is better" in doc


@pytest.mark.skipif(
    DISK_LASER_SUPPORTED,
    reason="disk_laser construction is allowed on supported builds; the unsupported-build "
    "rejection contract is pinned in test_disk_collection_dispatch.py and "
    "test_disk_collection_laser.py::test_disk_laser_unsupported_platform",
)
def test_disk_collection_index_type_rejection_disk_laser_unsupported_only(tmp_path):
    """On unsupported builds the constructor SHALL still raise ValueError on disk_laser."""
    path = str(tmp_path / "coll_laser")
    with pytest.raises(ValueError) as exc_info:
        DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_laser")
    msg = str(exc_info.value)
    assert "disk_laser" in msg
    assert "not implemented in v1" in msg
    assert not (tmp_path / "coll_laser").exists()


def test_disk_collection_index_type_rejection_unknown(tmp_path):
    path = str(tmp_path / "coll_unknown")
    with pytest.raises(ValueError) as exc_info:
        DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_unknown")
    msg = str(exc_info.value)
    assert "disk_unknown" in msg
    assert "disk_flat" in msg
    assert "disk_vamana" in msg
    assert not (tmp_path / "coll_unknown").exists()


def test_disk_collection_constructor_existing_path_raises(tmp_path):
    path = tmp_path / "coll"
    path.mkdir()
    with pytest.raises(RuntimeError):
        DiskCollection(path=str(path), dim=4, metric=MetricType.L2, index_type="disk_flat")


def test_disk_collection_open_missing_path_raises(tmp_path):
    with pytest.raises(RuntimeError):
        DiskCollection.open(str(tmp_path / "definitely_missing"))


def test_disk_collection_search_before_flush_returns_empty(tmp_path):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_flat")
    q = np.zeros(4, dtype=np.float32)
    hits = col.search(q, k=5)
    assert hits == []

    col.add(_rand_vectors(3, 4), _ids(3))
    hits2 = col.search(q, k=5)
    assert hits2 == [], "pending must be excluded from search"


def test_disk_collection_size_excludes_pending(tmp_path):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_flat")
    assert col.size() == 0
    col.add(_rand_vectors(7, 4), _ids(7))
    assert col.size() == 0  # pending excluded
    col.flush()
    assert col.size() == 7


def test_disk_collection_top_k_zero_raises(tmp_path):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_flat")
    col.add(_rand_vectors(3, 4), _ids(3))
    col.flush()
    q = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError):
        col.search(q, k=0)


def test_disk_collection_search_ef_zero_raises(tmp_path):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_flat")
    col.add(_rand_vectors(3, 4), _ids(3))
    col.flush()
    q = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        col.search(q, k=1, ef=0)
    assert "ef" in str(exc_info.value)


def test_disk_collection_top_k_exceeds_count_caps(tmp_path):
    path = str(tmp_path / "coll")
    col = DiskCollection(path=path, dim=4, metric=MetricType.L2, index_type="disk_flat")
    col.add(_rand_vectors(5, 4), _ids(5))
    col.flush()
    q = np.zeros(4, dtype=np.float32)
    hits = col.search(q, k=100)
    assert len(hits) == 5  # capped at count, no throw
