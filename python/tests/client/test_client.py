# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Unit tests for the AlayaLite Client class.
"""

# pylint: disable=protected-access

import gc
import os
import shutil
import tempfile
import unittest

import numpy as np
from alayalite import Client, Collection, Index


class _FakeClosable:
    def __init__(self, fail=False):
        self.fail = fail
        self.closed = False

    def close(self):
        self.closed = True
        if self.fail:
            raise RuntimeError("close failed")


class _FakeNativeIndex:
    def __init__(self):
        self.closed = False

    def close_db(self):
        self.closed = True


class TestClient(unittest.TestCase):
    """Test suite for client operations like creating, getting, and deleting collections and indices."""

    def setUp(self):
        # Create temp directory for RocksDB isolation
        self.tmp_dir = tempfile.mkdtemp()
        self._original_rocksdb_dir = os.environ.get("ALAYALITE_ROCKSDB_DIR")
        os.environ["ALAYALITE_ROCKSDB_DIR"] = os.path.join(self.tmp_dir, "RocksDB")
        self.client = Client()

    def tearDown(self):
        """Clean up temp directories after each test."""
        del self.client
        gc.collect()
        if self._original_rocksdb_dir is None:
            os.environ.pop("ALAYALITE_ROCKSDB_DIR", None)
        else:
            os.environ["ALAYALITE_ROCKSDB_DIR"] = self._original_rocksdb_dir
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_create_collection(self):
        collection = self.client.create_collection("test_collection")
        self.assertIsInstance(collection, Collection)
        self.assertIn("test_collection", self.client.list_collections())
        with self.assertRaises(RuntimeError):
            self.client.save_collection("test_collection")

    def test_create_duplicate_collection(self):
        self.client.create_collection("test_collection")
        with self.assertRaises(RuntimeError):
            self.client.create_collection("test_collection")

    def test_get_collection(self):
        self.client.create_collection("test_collection")
        collection = self.client.get_collection("test_collection")
        self.assertIsInstance(collection, Collection)

    def test_create_index(self):
        index = self.client.create_index("test_index", metric="ip")
        self.assertIsInstance(index, Index)
        self.assertIn("test_index", self.client.list_indices())
        with self.assertRaises(RuntimeError):
            self.client.save_index("test_index")

    def test_create_duplicate_index(self):
        self.client.create_index("test_index")
        with self.assertRaises(RuntimeError):
            self.client.create_index("test_index")

    def test_get_index(self):
        self.client.create_index("test_index")
        index = self.client.get_index("test_index")
        self.assertIsInstance(index, Index)

    def test_get_or_create_collection(self):
        collection1 = self.client.get_or_create_collection("test_collection")
        collection2 = self.client.get_or_create_collection("test_collection")
        self.assertIs(collection1, collection2)

    def test_get_or_create_index(self):
        index1 = self.client.get_or_create_index("test_index")
        index2 = self.client.get_or_create_index("test_index")
        self.assertIs(index1, index2)

    def test_delete_collection(self):
        self.client.create_collection("test_collection")
        with self.assertRaises(RuntimeError):  # Without url
            self.client.delete_collection("test_collection", True)
        self.assertNotIn("test_collection", self.client.list_collections())
        with self.assertRaises(RuntimeError):
            self.client.delete_collection("non_exist")

    def test_delete_index(self):
        self.client.create_index("test_index")
        with self.assertRaises(RuntimeError):  # Without url
            self.client.delete_index("test_index", True)
        self.assertNotIn("test_index", self.client.list_indices())
        with self.assertRaises(RuntimeError):
            self.client.delete_index("non_exist")

    def test_reset(self):
        self.client.create_collection("test_collection")
        self.client.create_index("test_index")
        self.client.reset()
        self.assertEqual(len(self.client.list_collections()), 0)
        self.assertEqual(len(self.client.list_indices()), 0)

    def test_reset_closes_managed_resources_and_deletes_disk_entries(self):
        url = os.path.join(self.tmp_dir, "client_url")
        os.makedirs(os.path.join(url, "collection_on_disk"))
        os.makedirs(os.path.join(url, "index_on_disk"))

        client = Client(url)
        collection = _FakeClosable()
        index = _FakeClosable()
        client._Client__collection_map = {"collection_on_disk": collection}
        client._Client__index_map = {"index_on_disk": index}

        client.reset(delete_on_disk=True)

        self.assertTrue(collection.closed)
        self.assertTrue(index.closed)
        self.assertFalse(os.path.exists(os.path.join(url, "collection_on_disk")))
        self.assertFalse(os.path.exists(os.path.join(url, "index_on_disk")))
        self.assertEqual(client.list_collections(), [])
        self.assertEqual(client.list_indices(), [])

    def test_close_helpers_ignore_native_close_errors(self):
        collection = _FakeClosable(fail=True)
        index = _FakeClosable(fail=True)

        Client._close_collection(collection)
        Client._close_index(index)

        self.assertTrue(collection.closed)
        self.assertTrue(index.closed)

    def test_index_close_releases_native_handle_once(self):
        index = Index("closable_index")
        native = _FakeNativeIndex()
        index._Index__index = native

        index.close()
        index.close()

        self.assertTrue(native.closed)
        self.assertIsNone(index._Index__index)

    def test_get_non_exist(self):
        index = self.client.get_index("non_exist")
        self.assertIsNone(index)
        coll = self.client.get_collection("non_exist")
        self.assertIsNone(coll)

    def test_dup_ind_coll(self):
        _ = self.client.create_index("dup", metric="cosine", quantization_type=None)
        with self.assertRaises(RuntimeError):
            _ = self.client.create_collection("dup")

    def test_dup_coll_ind(self):
        _ = self.client.create_collection("dup")
        with self.assertRaises(RuntimeError):
            _ = self.client.create_index("dup")

    def test_collection_params(self):
        items = [
            (
                1,
                "Document 1",
                np.array([0.1, 0.2, 0.3], dtype=float),
                {"category": "A"},
            ),
            (
                2,
                "Document 2",
                np.array([0.4, 0.5, 0.6], dtype=float),
                {"category": "B"},
            ),
        ]
        col1 = self.client.create_collection("col1")
        col1.insert(items)
        params1 = col1.get_index_params()
        self.assertEqual(params1.quantization_type, "none")

        col2 = self.client.create_collection("col2", quantization_type="sq8")
        col2.insert(items)
        params2 = col2.get_index_params()
        self.assertEqual(params2.quantization_type, "sq8")


if __name__ == "__main__":
    unittest.main()
