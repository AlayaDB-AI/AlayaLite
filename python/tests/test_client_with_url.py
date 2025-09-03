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

import os
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

    def test_save_and_delete_ind(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = Client(temp_dir)
            with self.assertRaises(RuntimeError):
                client.save_index("non-exist")
            index_name = "ind"
            index = client.create_index(index_name)
            index.fit(self.ind_vectors)
            self.assertIn(index_name, client.list_indices())

            client.save_index(index_name)
            index_path = os.path.join(temp_dir, index_name)
            self.assertTrue(os.path.isdir(index_path))
            schema_path = os.path.join(index_path, "schema.json")
            self.assertTrue(os.path.isfile(schema_path))

            client.delete_index(index_name, True)
            self.assertNotIn(index_name, client.list_indices())
            self.assertFalse(os.path.isdir(index_path))
            self.assertFalse(os.path.isfile(schema_path))

    def test_save_and_delete_coll(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = Client(temp_dir)
            with self.assertRaises(RuntimeError):
                client.save_collection("non-exist")
            coll_name = "coll"
            coll = client.create_collection(coll_name)
            coll.insert(self.coll_items)
            self.assertIn(coll_name, client.list_collections())

            client.save_collection(coll_name)
            coll_path = os.path.join(temp_dir, coll_name)
            self.assertTrue(os.path.isdir(coll_path))
            schema_path = os.path.join(coll_path, "schema.json")
            self.assertTrue(os.path.isfile(schema_path))

            client.delete_collection(coll_name, True)
            self.assertNotIn(coll_name, client.list_collections())
            self.assertFalse(os.path.isdir(coll_path))
            self.assertFalse(os.path.isfile(schema_path))

    def test_init_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = Client(temp_dir)
            index_name = "ind"
            index = client.create_index(
                index_name, metric="ip", quantization_type="sq8", index_type="nsg", max_nbrs=100
            )
            index.fit(self.ind_vectors)
            client.save_index(index_name)
            coll_name = "coll"
            coll = client.create_collection(coll_name)
            coll.insert(self.coll_items)
            client.save_collection(coll_name)
            _ = Client(temp_dir)  # load

    def test_load_different(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = Client(temp_dir)
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
            _ = Client(temp_dir)  # load

    def test_data_type_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = Client(temp_dir)
            index_name = "ind"
            index = client.create_index(index_name, data_type=np.int8)
            with self.assertRaises(ValueError):
                index.fit(self.ind_vectors)


if __name__ == "__main__":
    unittest.main()
