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

import unittest

from alayalite import Client, Collection, Index


class TestClient(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_create_collection(self):
        collection = self.client.create_collection("test_collection")
        self.assertIsInstance(collection, Collection)
        self.assertIn("test_collection", self.client.list_collections())

    def test_create_duplicate_collection(self):
        self.client.create_collection("test_collection")
        with self.assertRaises(RuntimeError):
            self.client.create_collection("test_collection")

    def test_get_collection(self):
        self.client.create_collection("test_collection")
        collection = self.client.get_collection("test_collection")
        self.assertIsInstance(collection, Collection)

    def test_create_index(self):
        index = self.client.create_index("test_index", index_type="flat")
        self.assertIsInstance(index, Index)
        self.assertIn("test_index", self.client.list_indices())

    def test_create_duplicate_index(self):
        self.client.create_index("test_index", index_type="flat")
        with self.assertRaises(RuntimeError):
            self.client.create_index("test_index", index_type="flat")

    def test_get_index(self):
        self.client.create_index("test_index", index_type="flat")
        index = self.client.get_index("test_index")
        self.assertIsInstance(index, Index)

    def test_get_or_create_collection(self):
        collection1 = self.client.get_or_create_collection("test_collection")
        collection2 = self.client.get_or_create_collection("test_collection")
        self.assertIs(collection1, collection2)

    def test_get_or_create_index(self):
        index1 = self.client.get_or_create_index("test_index", index_type="flat")
        index2 = self.client.get_or_create_index("test_index", index_type="flat")
        self.assertIs(index1, index2)

    def test_delete_collection(self):
        self.client.create_collection("test_collection")
        self.client.delete_collection("test_collection")
        self.assertNotIn("test_collection", self.client.list_collections())

    def test_delete_index(self):
        self.client.create_index("test_index", index_type="flat")
        self.client.delete_index("test_index")
        self.assertNotIn("test_index", self.client.list_indices())

    def test_reset(self):
        self.client.create_collection("test_collection")
        self.client.create_index("test_index", index_type="flat")
        self.client.reset()
        self.assertEqual(len(self.client.list_collections()), 0)
        self.assertEqual(len(self.client.list_indices()), 0)


if __name__ == "__main__":
    unittest.main()
