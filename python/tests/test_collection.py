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

import random
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

    def test_batch_query1(self):
        items = [(1, "Document 1", np.array([0.1, 0.2, 0.3]), {}), (2, "Document 2", np.array([0.4, 0.5, 0.6]), {})]
        self.collection.insert(items)
        result = self.collection.batch_query([[0.1, 0.2, 0.3]], limit=1, ef_search=10, num_threads=1)
        self.assertEqual(len(result["id"]), 1)

    def test_batch_query2(self):
        items = [
            (1, "document1", [1.0, 2.0, 3.0], {}),
            (2, "document2", [4.0, 5.0, 6.0], {"category": "B"}),
            (3, "document3", [12.0, 32.0, 31.0], {"category": "C"}),
        ]
        self.collection.insert(items)

        result = self.collection.batch_query([[12.0, 32.0, 31.0]], limit=1, ef_search=10, num_threads=1)
        self.assertEqual(len(result["id"][0]), 1)
        self.assertEqual(int(result["id"][0][0]), 3)

        result = self.collection.batch_query(
            [[12.0, 32.0, 31.0], [4.0, 5.0, 6.0]], limit=3, ef_search=10, num_threads=1
        )
        self.assertEqual(len(result["id"][0]), 3)
        self.assertEqual(len(result["id"][1]), 3)
        self.assertEqual(list(map(int, result["id"][0])), [3, 2, 1])
        self.assertEqual(list(map(int, result["id"][1])), [2, 1, 3])

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

    def test_get_by_id(self):
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]), {"category": "B"}),
        ]
        self.collection.insert(items)
        result = self.collection.get_by_id([1])
        self.assertEqual(len(result["id"]), 1)
        self.assertEqual(result["document"][0], "Document 1")

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

    def test_mixed_query(self):
        uuids = [f"id {i}" for i in range(1, 1000)]
        documents = [f"Document {i}" for i in range(1, 1000)]
        embeddings = [[random.random() for _ in range(100)] for _ in range(1000)]
        self.collection.insert(list(zip(uuids, documents, embeddings, [{} for _ in range(1000)])))
        self.collection.delete_by_id(uuids[10:500:2])
        self.collection.insert(
            list(zip(uuids[10:500:2], documents[10:500:2], embeddings[10:500:2], [{} for _ in range(500)]))
        )

        for uid, document, embedding in zip(uuids, documents, embeddings):
            result = self.collection.batch_query([embedding], limit=10, ef_search=50, num_threads=1)
            self.assertEqual(result["id"][0][0], uid)
            self.assertEqual(result["document"][0][0], document)
            self.assertEqual(result["distance"][0][0], 0.0)


if __name__ == "__main__":
    unittest.main()
