"""Tests for external-ID based DiskANN update operations."""

import sys

import numpy as np
import pytest
from alayalite import diskann

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="DiskANN updates require Linux")


def _build_index(path):
    rng = np.random.default_rng(42)
    vectors = rng.random((96, 16), dtype=np.float32)
    external_ids = np.arange(1000, 1096, dtype=np.uint64)
    build_params = diskann.BuildParams()
    build_params.R = 16
    build_params.L = 32
    diskann.Index.build(str(path), vectors, external_ids, build_params)

    load_params = diskann.LoadParams()
    load_params.updatable = True
    load_params.update_io = diskann.UpdateIO.BLOCKING
    load_params.num_threads = 2
    load_params.update_insert_threads = 2
    load_params.update_reconnect_threads = 2
    return diskann.Index.open(str(path), load_params), vectors


def test_update_uses_external_ids_only(tmp_path):
    index, _ = _build_index(tmp_path / "index")
    vector = np.random.default_rng(7).random(16, dtype=np.float32)

    index.insert(vector, 5000)
    assert index.contains(5000)
    labels, _ = index.search(vector, 5)
    assert 5000 in labels

    index.remove(5000)
    assert not index.contains(5000)
    with pytest.raises(KeyError, match="5000"):
        index.remove(5000)


def test_batch_update_validates_labels_before_mutation(tmp_path):
    index, _ = _build_index(tmp_path / "index")
    vectors = np.random.default_rng(8).random((2, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="duplicate external label"):
        index.batch_insert(vectors, np.array([6000, 6000], dtype=np.uint64))
    assert not index.contains(6000)

    index.batch_insert(vectors, np.array([6000, 6001], dtype=np.uint64))
    with pytest.raises(KeyError, match="999999"):
        index.batch_remove(np.array([6000, 999999], dtype=np.uint64))
    assert index.contains(6000)
    assert index.contains(6001)

    index.batch_remove(np.array([6000, 6001], dtype=np.uint64))
    assert not index.contains(6000)
    assert not index.contains(6001)


def test_external_id_mapping_survives_flush_and_reload(tmp_path):
    path = tmp_path / "index"
    index, _ = _build_index(path)
    vector = np.random.default_rng(9).random(16, dtype=np.float32)
    index.insert(vector, 7000)
    index.remove(1001)
    index.flush()
    del index

    load_params = diskann.LoadParams()
    load_params.updatable = True
    load_params.update_io = diskann.UpdateIO.BLOCKING
    reopened = diskann.Index.open(str(path), load_params)
    assert reopened.contains(7000)
    assert not reopened.contains(1001)
    reopened.remove(7000)


def test_update_arrays_require_exact_dtype_and_layout(tmp_path):
    index, _ = _build_index(tmp_path / "index")

    with pytest.raises(TypeError, match="float32"):
        index.insert(np.ones(16, dtype=np.float64), 8000)
    with pytest.raises(TypeError, match="uint64"):
        index.batch_remove(np.array([1001], dtype=np.int64))
