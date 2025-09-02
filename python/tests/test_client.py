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

"""
Unit tests for the AlayaLite Client class.
"""

import unittest

from alayalite import Client, Collection, Index


class TestClient(unittest.TestCase):
    """Test suite for client operations like creating, getting, and deleting collections and indices."""

    def setUp(self):
        self.client = Client()

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


if __name__ == "__main__":
    unittest.main()
