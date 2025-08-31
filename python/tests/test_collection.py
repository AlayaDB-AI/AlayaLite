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
from alayalite.utils import calc_gt, calc_recall


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

    def test_reindex_large_scale(self):
        """Large-scale reindex test with recall evaluation.

        This test verifies that the Collection class maintains high recall after:
        1. Building the initial index.
        2. Deleting a large portion of items (90%).
        3. Performing a reindex on the remaining items.
        """

        dim = 128
        total = 1000

        # Step 1: Insert 1000 random vectors
        ids = list(range(total))
        vectors = np.random.rand(total, dim).astype(np.float32)
        docs = [f"Doc {i}" for i in ids]
        metas = [{} for _ in ids]
        self.collection.insert(list(zip(ids, docs, vectors, metas)))

        # --- Initial recall check immediately after building the index ---
        queries = vectors[np.random.choice(total, 20, replace=False)]  # 20 random queries
        result = self.collection.batch_query(queries.tolist(), limit=10, ef_search=100)
        retrieved = np.array(result["id"], dtype=int)
        gt = calc_gt(vectors, queries, 10)
        recall = calc_recall(retrieved, gt)
        self.assertGreaterEqual(recall, 0.9, f"Initial recall too low: {recall}")

        # Step 2: Delete 90% (first 900 ids), then reindex
        deleted_ids = ids[:900]
        remaining_ids = ids[900:]
        self.collection.delete_by_id(deleted_ids)

        # Verify deleted are gone
        result_deleted = self.collection.get_by_id(deleted_ids)
        self.assertEqual(len(result_deleted["id"]), 0)

        # Verify remaining are still there
        result_remaining = self.collection.get_by_id(remaining_ids)
        self.assertEqual(set(result_remaining["id"]), set(remaining_ids))

        # --- Recall check after deleting 90% ---
        remaining_vectors = vectors[900:]  # 100 remaining vectors
        queries_after_delete = remaining_vectors[np.random.choice(len(remaining_vectors), 20, replace=True)]
        result_before_reindex = self.collection.batch_query(queries_after_delete.tolist(), limit=10, ef_search=100)
        id_before_reindex = np.array(result_before_reindex["id"], dtype=int)
        gt_before_reindex = calc_gt(remaining_vectors, queries_after_delete, 10)
        recall_before_reindex = calc_recall(np.array([each - 900 for each in id_before_reindex]), gt_before_reindex)
        print(f"Recall after deleting 90% of points (before reindex): {recall_before_reindex:.4f}")

        self.collection.reindex()

        result_after_reindex = self.collection.batch_query(queries_after_delete.tolist(), limit=10, ef_search=100)
        retrieved_after_reindex = np.array(result_after_reindex["id"], dtype=int)
        gt_after_reindex = calc_gt(remaining_vectors, queries_after_delete, 10)
        recall_after_reindex = calc_recall(np.array([each - 900 for each in retrieved_after_reindex]), gt_after_reindex)
        print(f"Recall after deleting 90% of points (after reindex): {recall_after_reindex:.4f}")

        self.assertGreaterEqual(
            recall_after_reindex,
            recall_before_reindex,
            f"Recall decreased after reindex: before={recall_before_reindex:.4f}, after={recall_after_reindex:.4f}",
        )
        self.assertGreaterEqual(recall_after_reindex, 0.9, f"Recall too low after reindex: {recall}")


if __name__ == "__main__":
    unittest.main()
