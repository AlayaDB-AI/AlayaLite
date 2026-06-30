# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Unit tests for the update functionality of the AlayaLite Index,
such as inserting vectors and handling capacity limits.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from alayalite import Client
from alayalite.index import Index
from alayalite.schema import IndexParams


class TestAlayaLiteUpdate(unittest.TestCase):
    """Test suite for index update operations."""

    def setUp(self):
        """Set up a new client for each test."""
        self.client = Client()

    def test_insert_vector(self):
        """Test inserting new vectors into a fitted index."""
        index = self.client.create_index()
        vectors = np.random.rand(1000, 128).astype(np.float32)
        index.fit(vectors)

        new_vector_1 = np.random.rand(128).astype(np.float32)
        new_id_1 = index.insert(new_vector_1)
        self.assertEqual(new_id_1, 1000)

        new_vector_2 = np.random.rand(128).astype(np.float32)
        new_id_2 = index.insert(new_vector_2)
        self.assertEqual(new_id_2, 1001)

        vector_1 = index.get_data_by_id(1000)
        self.assertTrue(np.allclose(vector_1, new_vector_1))

        vector_2 = index.get_data_by_id(1001)
        self.assertTrue(np.allclose(vector_2, new_vector_2))

    def test_insert_accepts_list_vector(self):
        index = self.client.create_index()
        index.fit(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

        new_id = index.insert([4.0, 5.0, 6.0])

        self.assertEqual(new_id, 1)
        self.assertTrue(np.allclose(index.get_data_by_id(1), np.array([4.0, 5.0, 6.0], dtype=np.float32)))

    def test_fit_failure_leaves_index_retryable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rocksdb_path = os.path.join(tmp_dir, "rocksdb")
            index = Index(
                "retryable_index",
                IndexParams(
                    rocksdb_path=rocksdb_path,
                    has_scalar_data=True,
                ),
            )
            vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
            with self.assertRaisesRegex(RuntimeError, "Duplicate item_id: dup"):
                index.fit(
                    vectors,
                    item_ids=["dup", "dup"],
                    documents=["Document 1", "Document 2"],
                    metadata_list=[{}, {}],
                )

            self.assertFalse(os.path.exists(rocksdb_path))

            index.fit(
                vectors[:1],
                item_ids=["ok"],
                documents=["Document OK"],
                metadata_list=[{}],
            )
            self.assertEqual(index.search([1.0, 2.0, 3.0], 1)[0], 0)

    def test_fit_failure_removes_created_rocksdb_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rocksdb_path = os.path.join(tmp_dir, "rocksdb")

            class FailingNativeIndex:
                """Native index double that creates RocksDB files before fit fails."""

                def __init__(self, params):
                    self.params = params
                    self.closed = False

                def fit(self, *_args):
                    os.makedirs(rocksdb_path)
                    with open(os.path.join(rocksdb_path, "orphan"), "w", encoding="utf-8") as f:
                        f.write("orphaned scalar data")
                    raise RuntimeError("native fit failed")

                def close_db(self):
                    self.closed = True

            index = Index(
                "failed_index",
                IndexParams(rocksdb_path=rocksdb_path, has_scalar_data=True),
            )

            with patch("alayalite.index._PyIndexInterface", FailingNativeIndex):
                with self.assertRaisesRegex(RuntimeError, "native fit failed"):
                    index.fit(
                        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                        item_ids=["item"],
                        documents=["Document"],
                        metadata_list=[{}],
                    )

            self.assertFalse(os.path.exists(rocksdb_path))

    def test_fit_cleanup_preserves_original_error_when_close_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rocksdb_path = os.path.join(tmp_dir, "rocksdb")

            class CloseFailingNativeIndex:
                """Native index double whose close path fails after fit fails."""

                def __init__(self, params):
                    self.params = params

                def fit(self, *_args):
                    os.makedirs(rocksdb_path)
                    raise RuntimeError("native fit failed")

                def close_db(self):
                    raise RuntimeError("close failed")

            index = Index(
                "close_failed_index",
                IndexParams(rocksdb_path=rocksdb_path, has_scalar_data=True),
            )

            with patch("alayalite.index._PyIndexInterface", CloseFailingNativeIndex):
                with self.assertRaisesRegex(RuntimeError, "native fit failed") as raised:
                    index.fit(
                        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                        item_ids=["item"],
                        documents=["Document"],
                        metadata_list=[{}],
                    )

            notes = getattr(raised.exception, "__notes__", [])
            cleanup_details = "\n".join(notes) or str(raised.exception)
            self.assertIn("close_db failed during failed fit cleanup", cleanup_details)
            self.assertFalse(os.path.exists(rocksdb_path))

    def test_fit_failure_keeps_preexisting_rocksdb_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rocksdb_path = os.path.join(tmp_dir, "rocksdb")
            os.makedirs(rocksdb_path)
            existing_file = os.path.join(rocksdb_path, "existing")
            with open(existing_file, "w", encoding="utf-8") as f:
                f.write("existing scalar data")

            class FailingNativeIndex:
                """Native index double that fails after seeing existing RocksDB data."""

                def __init__(self, params):
                    self.params = params

                def fit(self, *_args):
                    with open(os.path.join(rocksdb_path, "orphan"), "w", encoding="utf-8") as f:
                        f.write("new failed fit data")
                    raise RuntimeError("native fit failed")

                def close_db(self):
                    pass

            index = Index(
                "preexisting_path_index",
                IndexParams(rocksdb_path=rocksdb_path, has_scalar_data=True),
            )

            with patch("alayalite.index._PyIndexInterface", FailingNativeIndex):
                with self.assertRaisesRegex(RuntimeError, "native fit failed"):
                    index.fit(
                        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                        item_ids=["item"],
                        documents=["Document"],
                        metadata_list=[{}],
                    )

            self.assertTrue(os.path.exists(existing_file))

    def test_index_out_of_scope(self):
        """Test that inserting into a full index raises a RuntimeError."""
        index = self.client.create_index(capacity=1000)
        vectors = np.random.rand(1000, 128).astype(np.float32)
        index.fit(vectors)

        new_vector_1 = np.random.rand(128).astype(np.float32)
        with self.assertRaises(RuntimeError):
            index.insert(new_vector_1)


if __name__ == "__main__":
    unittest.main()
