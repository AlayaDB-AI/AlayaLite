"""Tests for Collection.get_embeddings_by_id."""

import numpy as np
import pytest
from alayalite.collection import Collection


def test_get_embeddings_by_id_roundtrip():
    col = Collection("mmr_test_collection")

    items = [
        ("a", "doc-a", [1.0, 0.0, 0.0], {"k": 1}),
        ("b", "doc-b", [0.0, 1.0, 0.0], {"k": 2}),
        ("c", "doc-c", [0.0, 0.0, 1.0], {"k": 3}),
    ]
    col.insert(items)

    got = col.get_embeddings_by_id(["b", "a", "c"])
    assert len(got) == 3
    assert np.allclose(got[0], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert np.allclose(got[1], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(got[2], np.array([0.0, 0.0, 1.0], dtype=np.float32))


def test_get_embeddings_by_id_missing_raises():
    col = Collection("mmr_test_collection_missing")
    col.insert([("a", "doc-a", [1.0, 0.0], {})])

    with pytest.raises(KeyError):
        col.get_embeddings_by_id(["a", "not-exist"])
