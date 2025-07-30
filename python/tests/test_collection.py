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
Unit tests for the AlayaLite Collection class.
"""

import random
import unittest

import numpy as np

from alayalite import Collection


class TestCollection(unittest.TestCase):
    """Test suite for Collection class operations."""

    def setUp(self):
        """Set up a new collection for each test."""
        self.collection = Collection("test_collection")

    def test_insert(self):
        """Test inserting items into the collection."""
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]), {"category": "B"}),
        ]
        self.collection.insert(items)
        result = self.collection.filter_query({})
        self.assertEqual(len(result["id"]), 2)

    def test_batch_query_single_result(self):
        """Test a batch query that returns a single nearest neighbor."""
        items = [(1, "Doc 1", np.array([0.1, 0.2, 0.3]), {}), (2, "Doc 2", np.array([0.4, 0.5, 0.6]), {})]
        self.collection.insert(items)
        result = self.collection.batch_query([[0.1, 0.2, 0.3]], limit=1, ef_search=10)
        self.assertEqual(len(result["id"]), 1)
        self.assertEqual(result["id"][0][0], 1)

    def test_batch_query_multiple_results(self):
        """Test a batch query with multiple queries and expected neighbors."""
        items = [
            (1, "document1", [1.0, 2.0, 3.0], {}),
            (2, "document2", [4.0, 5.0, 6.0], {"category": "B"}),
            (3, "document3", [12.0, 32.0, 31.0], {"category": "C"}),
        ]
        self.collection.insert(items)

        result_single = self.collection.batch_query([[12.0, 32.0, 31.0]], limit=1, ef_search=10)
        self.assertEqual(len(result_single["id"][0]), 1)
        self.assertEqual(int(result_single["id"][0][0]), 3)

        result_multi = self.collection.batch_query([[12.0, 32.0, 31.0], [4.0, 5.0, 6.0]], limit=3, ef_search=10)
        self.assertEqual(len(result_multi["id"][0]), 3)
        self.assertEqual(len(result_multi["id"][1]), 3)
        self.assertEqual(list(map(int, result_multi["id"][0])), [3, 2, 1])
        self.assertEqual(list(map(int, result_multi["id"][1])), [2, 1, 3])

    def test_upsert(self):
        """Test updating an existing item in the collection."""
        items = [(1, "Old Doc", np.array([0.1, 0.2, 0.3]), {})]
        self.collection.insert(items)
        update_items = [(1, "New Doc", np.array([0.2, 0.3, 0.4]), {})]
        self.collection.upsert(update_items)
        result = self.collection.get_by_id([1])
        self.assertEqual(len(result["document"]), 1)
        self.assertEqual(result["document"][0], "New Doc")

    def test_delete_by_id(self):
        """Test deleting an item by its ID."""
        items = [(1, "Document 1", np.array([0.1, 0.2, 0.3]), {})]
        self.collection.insert(items)
        self.collection.delete_by_id([1])
        result = self.collection.get_by_id([1])
        self.assertEqual(len(result["id"]), 0)

    def test_get_by_id(self):
        """Test retrieving items by their IDs."""
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]), {"category": "B"}),
        ]
        self.collection.insert(items)
        result = self.collection.get_by_id([1])
        self.assertEqual(len(result["id"]), 1)
        self.assertEqual(result["document"][0], "Document 1")

    def test_delete_by_filter(self):
        """Test deleting items based on a metadata filter."""
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]), {"category": "A"}),
            (3, "Document 3", np.array([0.7, 0.8, 0.9]), {"category": "B"}),
        ]
        self.collection.insert(items)
        self.collection.delete_by_filter({"category": "A"})
        result = self.collection.filter_query({})
        self.assertEqual(len(result["id"]), 1)
        self.assertEqual(result["id"][0], 3)

    def test_filter_query(self):
        """Test querying items based on a metadata filter."""
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]), {"category": "B"}),
        ]
        self.collection.insert(items)
        result = self.collection.filter_query({"category": "A"})
        self.assertEqual(len(result["id"]), 1)
        self.assertEqual(result["document"][0], "Document 1")

    def test_mixed_operations(self):
        """Test a sequence of insert, delete, and query operations."""
        uuids = [f"id_{i}" for i in range(1, 1000)]
        documents = [f"Document {i}" for i in range(1, 1000)]
        embeddings = [[random.random() for _ in range(100)] for _ in range(999)]
        # Ensure all lists have the same length
        self.collection.insert(list(zip(uuids, documents, embeddings, [{} for _ in range(999)])))

        ids_to_delete = uuids[10:500:2]
        self.collection.delete_by_id(ids_to_delete)

        # Verify deletion
        result_after_delete = self.collection.get_by_id(ids_to_delete)
        self.assertEqual(len(result_after_delete["id"]), 0)

        # Test query on a remaining item
        test_index = 5
        result = self.collection.batch_query([embeddings[test_index]], limit=10, ef_search=50)
        self.assertEqual(result["id"][0][0], uuids[test_index])
        self.assertEqual(result["document"][0][0], documents[test_index])
        self.assertAlmostEqual(result["distance"][0][0], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
