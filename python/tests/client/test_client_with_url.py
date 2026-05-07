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

"""Tests for Client class functionality with URL-based operations."""

import gc
import os
import shutil
import tempfile
import unittest

import numpy as np
from alayalite import Client


class TestClientWithURL(unittest.TestCase):
    """Test cases for Client class operations with URL-based storage."""

    def setUp(self):
        self.ind_vectors = np.random.rand(1000, 128).astype(np.float32)
        self.coll_items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]), {"category": "B"}),
        ]
        # Create temp directory for test isolation
        self.temp_dir = tempfile.mkdtemp()
        # Set RocksDB directory to temp_dir for isolation
        self._original_rocksdb_dir = os.environ.get("ALAYALITE_ROCKSDB_DIR")
        os.environ["ALAYALITE_ROCKSDB_DIR"] = os.path.join(self.temp_dir, "RocksDB")
        self.client = Client(self.temp_dir)

    def tearDown(self):
        """Clean up temp directories after each test."""
        del self.client
        gc.collect()
        if self._original_rocksdb_dir is None:
            os.environ.pop("ALAYALITE_ROCKSDB_DIR", None)
        else:
            os.environ["ALAYALITE_ROCKSDB_DIR"] = self._original_rocksdb_dir
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_and_delete_ind(self):
        client = Client(self.temp_dir)
        with self.assertRaises(RuntimeError):
            client.save_index("non-exist")
        index_name = "ind"
        index = client.create_index(index_name)
        index.fit(self.ind_vectors)
        self.assertIn(index_name, client.list_indices())

        client.save_index(index_name)
        index_path = os.path.join(self.temp_dir, index_name)
        self.assertTrue(os.path.isdir(index_path))
        schema_path = os.path.join(index_path, "schema.json")
        self.assertTrue(os.path.isfile(schema_path))

        client.delete_index(index_name, True)
        self.assertNotIn(index_name, client.list_indices())
        self.assertFalse(os.path.isdir(index_path))
        self.assertFalse(os.path.isfile(schema_path))

    def test_save_and_delete_coll(self):
        client = Client(self.temp_dir)
        with self.assertRaises(RuntimeError):
            client.save_collection("non-exist")
        coll_name = "coll"
        coll = client.create_collection(coll_name)
        self.assertEqual(coll.get_index_params().rocksdb_path, os.path.join(self.temp_dir, coll_name, "rocksdb"))
        coll.insert(self.coll_items)
        self.assertIn(coll_name, client.list_collections())

        client.save_collection(coll_name)
        coll_path = os.path.join(self.temp_dir, coll_name)
        self.assertTrue(os.path.isdir(coll_path))
        schema_path = os.path.join(coll_path, "schema.json")
        self.assertTrue(os.path.isfile(schema_path))

        client.delete_collection(coll_name, True)
        self.assertNotIn(coll_name, client.list_collections())
        self.assertFalse(os.path.isdir(coll_path))
        self.assertFalse(os.path.isfile(schema_path))

    def test_init_load(self):
        client = Client(self.temp_dir)
        index_name = "ind"
        index = client.create_index(index_name, metric="ip", quantization_type="sq8", index_type="nsg", max_nbrs=100)
        index.fit(self.ind_vectors)
        client.save_index(index_name)
        coll_name = "coll"
        coll = client.create_collection(coll_name)
        coll.insert(self.coll_items)
        client.save_collection(coll_name)
        # # Delete references to release RocksDB lock before loading
        del coll
        del index
        del client

        gc.collect()
        _ = Client(self.temp_dir)  # load

    def test_collection_recovers_without_explicit_save(self):
        client = Client(self.temp_dir)
        coll = client.create_collection("recovering")
        coll.insert(
            [
                ("a", "Document A", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"group": "keep"}),
                ("b", "Document B", np.array([0.0, 1.0, 0.0], dtype=np.float32), {"group": "drop"}),
            ]
        )
        coll.upsert(
            [
                ("a", "Document A v2", np.array([1.0, 0.1, 0.0], dtype=np.float32), {"group": "keep", "v": 2}),
                ("c", "Document C", np.array([0.0, 0.0, 1.0], dtype=np.float32), {"group": "keep"}),
            ]
        )
        coll.delete_by_id(["b"])

        del coll
        del client
        gc.collect()

        recovered_client = Client(self.temp_dir)
        recovered = recovered_client.get_collection("recovering")

        self.assertIsNotNone(recovered)
        result = recovered.get_by_id(["a", "b", "c"])
        self.assertEqual(result["id"], ["a", "c"])
        self.assertEqual(result["document"][0], "Document A v2")
        self.assertEqual(result["metadata"][0]["v"], 2)

    def test_sq8_collection_recovery_uses_scalar_storage(self):
        client = Client(self.temp_dir)
        coll = client.create_collection("recovering_sq8", quantization_type="sq8", metric="ip")
        coll.insert(
            [
                ("x", "Document X", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"kind": "seed"}),
                ("y", "Document Y", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"kind": "old"}),
            ]
        )
        coll.upsert([("y", "Document Y v2", np.array([0.4, 0.5, 0.7], dtype=np.float32), {"kind": "new"})])

        del coll
        del client
        gc.collect()

        recovered_client = Client(self.temp_dir)
        recovered = recovered_client.get_collection("recovering_sq8")

        self.assertIsNotNone(recovered)
        result = recovered.get_by_id(["x", "y"])
        self.assertEqual(result["id"], ["x", "y"])
        self.assertEqual(result["document"][1], "Document Y v2")
        self.assertEqual(result["metadata"][1]["kind"], "new")

    def test_load_different(self):
        client = Client(self.temp_dir)
        index_name = "ind1"
        index1 = client.create_index(
            index_name, metric="cos", quantization_type="sq4", index_type="fusion", id_type=np.uint32
        )
        index1.fit(self.ind_vectors)
        with self.assertRaises(RuntimeError):  # double fit
            index1.fit(self.ind_vectors)
        client.save_index(index_name)
        with self.assertRaises(ValueError):
            _ = client.create_index("illegal_nbr_ind", max_nbrs=1000)
        _ = Client(self.temp_dir)  # load

    def test_data_type_mismatch(self):
        client = Client(self.temp_dir)
        index_name = "ind"
        index = client.create_index(index_name, data_type=np.int8)
        with self.assertRaises(ValueError):
            index.fit(self.ind_vectors)


if __name__ == "__main__":
    unittest.main()
