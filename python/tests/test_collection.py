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

import numpy as np

from alayalite import Collection


class TestCollection(unittest.TestCase):
    def setUp(self):
        self.collection = Collection("test_collection")

    def test_insert(self):
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]), {"category": "B"}),
        ]
        self.collection.insert(items)
        result = self.collection.filter_query({})
        self.assertEqual(len(result["id"]), 2)

    def test_batch_query(self):
        items = [(1, "Document 1", np.array([0.1, 0.2, 0.3]), {}), (2, "Document 2", np.array([0.4, 0.5, 0.6]), {})]
        self.collection.insert(items)
        result = self.collection.batch_query([[0.1, 0.2, 0.3]], limit=1, ef_search=10, num_threads=1)
        self.assertEqual(len(result["id"]), 1)

    def test_upsert(self):
        items = [(1, "Old Doc", np.array([0.1, 0.2, 0.3]), {})]
        self.collection.insert(items)
        update_items = [(1, "New Doc", np.array([0.2, 0.3, 0.4]), {})]
        self.collection.upsert(update_items)
        result = self.collection.filter_query({})
        self.assertEqual(len(result["document"]), 1)
        self.assertEqual(result["document"][0], "New Doc")

    def test_delete_by_id(self):
        items = [(1, "Document 1", np.array([0.1, 0.2, 0.3]), {})]
        self.collection.insert(items)
        self.collection.delete_by_id([1])
        df = self.collection.filter_query({})
        self.assertEqual(len(df), 0)

    def test_delete_by_filter(self):
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]), {"category": "A"}),
            (3, "Document 3", np.array([0.7, 0.8, 0.9]), {"category": "B"}),
        ]
        self.collection.insert(items)
        self.collection.delete_by_filter({"category": "A"})
        df = self.collection.filter_query({})
        self.assertEqual(len(df["id"]), 1)
        self.assertEqual(df["id"][0], 3)

    def test_filter_query(self):
        # items = [
        #   (1, "Document 1", np.array([0.1, 0.2, 0.3]), {"category": "A"}),
        #   (2, "Document 2", np.array([0.4, 0.5, 0.6]), {"category": "B"}),
        # ]
        result = self.collection.filter_query({"category": "A"})
        print(result)
        # self.assertEqual(result["document"][0], "Document 1")


if __name__ == "__main__":
    unittest.main()
