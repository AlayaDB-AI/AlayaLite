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
Unit tests for @notify decorated functions with non-empty payload.
This test suite ensures all functions decorated with @notify properly handle payload.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from alayalite import Client


class TestNotifyIndex(unittest.TestCase):
    """Test suite for Index class @notify decorated methods with payload."""

    def setUp(self):
        """Set up a new client and index for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.client = Client(self.temp_dir)
        self.index = self.client.create_index("test_index")
        self.vectors = np.random.rand(100, 128).astype(np.float32)
        self.payload = {"test_key": "test_value"}

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_index_fit_with_payload(self):
        """Test Index.fit() with non-empty payload."""
        self.index.fit(self.vectors, ef_construction=100, num_threads=1, payload=self.payload)

        # Verify payload was updated with monitoring data
        self.assertIn("vector_dim", self.payload)
        self.assertIn("total_vectors", self.payload)
        self.assertIn("index_type", self.payload)
        self.assertEqual(self.payload["vector_dim"], 128)
        self.assertEqual(self.payload["total_vectors"], 100)

    def test_index_insert_with_payload(self):
        """Test Index.insert() with non-empty payload."""
        self.index.fit(self.vectors, ef_construction=100, num_threads=1)

        new_vector = np.random.rand(128).astype(np.float32)
        self.index.insert(new_vector, ef=100, payload=self.payload)

        # Verify payload was updated
        self.assertIn("vector_dim", self.payload)
        self.assertIn("index_type", self.payload)
        self.assertEqual(self.payload["vector_dim"], 128)

    def test_index_search_with_payload(self):
        """Test Index.search() with non-empty payload."""
        self.index.fit(self.vectors, ef_construction=100, num_threads=1)

        query = np.random.rand(128).astype(np.float32)
        results = self.index.search(query, topk=10, ef_search=100, payload=self.payload)

        # Verify payload was updated
        self.assertIn("vector_dim", self.payload)
        self.assertIn("index_type", self.payload)
        self.assertEqual(self.payload["vector_dim"], 128)
        self.assertIsNotNone(results)

    def test_index_batch_search_with_payload(self):
        """Test Index.batch_search() with non-empty payload."""
        self.index.fit(self.vectors, ef_construction=100, num_threads=1)

        queries = np.random.rand(10, 128).astype(np.float32)
        results = self.index.batch_search(queries, topk=5, ef_search=100, num_threads=1, payload=self.payload)

        # Verify payload was updated
        self.assertIn("vector_dim", self.payload)
        self.assertIn("index_type", self.payload)
        self.assertEqual(self.payload["vector_dim"], 128)
        self.assertIsNotNone(results)

    def test_index_batch_search_with_distance_and_payload(self):
        """Test Index.batch_search_with_distance() with non-empty payload."""
        self.index.fit(self.vectors, ef_construction=100, num_threads=1)

        queries = np.random.rand(10, 128).astype(np.float32)
        results = self.index.batch_search_with_distance(
            queries, topk=5, ef_search=100, num_threads=1, payload=self.payload
        )

        # Verify payload was updated
        self.assertIn("vector_dim", self.payload)
        self.assertIn("index_type", self.payload)
        self.assertEqual(self.payload["vector_dim"], 128)
        self.assertIsNotNone(results)

    def test_index_save_with_payload(self):
        """Test Index.save() with non-empty payload via Client."""
        self.index.fit(self.vectors, ef_construction=100, num_threads=1)

        self.client.save_index("test_index", payload=self.payload)

        # Verify payload was updated
        self.assertIn("vector_dim", self.payload)
        self.assertIn("index_type", self.payload)
        self.assertEqual(self.payload["vector_dim"], 128)

    def test_index_load_with_payload(self):
        """Test Index.load() with non-empty payload via Client."""
        self.index.fit(self.vectors, ef_construction=100, num_threads=1)
        self.client.save_index("test_index")

        # Create a new client to load the index
        new_client = Client(self.temp_dir)
        loaded_index = new_client.get_index("test_index")

        # Verify the index was loaded correctly
        self.assertIsNotNone(loaded_index)
        self.assertEqual(loaded_index.get_dim(), 128)


class TestNotifyCollection(unittest.TestCase):
    """Test suite for Collection class @notify decorated methods with payload."""

    def setUp(self):
        """Set up a new client and collection for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.client = Client(self.temp_dir)
        self.collection = self.client.create_collection("test_collection")
        self.items = [
            (1, "Document 1", [0.1, 0.2, 0.3], {"category": "A"}),
            (2, "Document 2", [0.4, 0.5, 0.6], {"category": "B"}),
            (3, "Document 3", [0.7, 0.8, 0.9], {"category": "C"}),
        ]
        self.payload = {"test_key": "test_value"}

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_collection_insert_with_payload(self):
        """Test Collection.insert() with non-empty payload."""
        self.collection.insert(self.items, payload=self.payload)

        # Verify payload was updated
        self.assertIn("collection_name", self.payload)
        self.assertIn("vector_dim", self.payload)
        self.assertIn("total_vectors", self.payload)
        self.assertEqual(self.payload["collection_name"], "test_collection")

    def test_collection_upsert_with_payload(self):
        """Test Collection.upsert() with non-empty payload."""
        self.collection.upsert(self.items, payload=self.payload)

        # Verify payload was updated
        self.assertIn("collection_name", self.payload)
        self.assertIn("vector_dim", self.payload)
        self.assertIn("total_vectors", self.payload)
        self.assertEqual(self.payload["collection_name"], "test_collection")

    def test_collection_batch_query_with_payload(self):
        """Test Collection.batch_query() with non-empty payload."""
        self.collection.insert(self.items)

        results = self.collection.batch_query(
            [[0.1, 0.2, 0.3]], limit=2, ef_search=10, num_threads=1, payload=self.payload
        )

        # Verify payload was updated
        self.assertIn("collection_name", self.payload)
        self.assertIn("vector_dim", self.payload)
        self.assertIn("total_vectors", self.payload)
        self.assertEqual(self.payload["collection_name"], "test_collection")
        self.assertIsNotNone(results)

    def test_collection_delete_by_id_with_payload(self):
        """Test Collection.delete_by_id() with non-empty payload."""
        self.collection.insert(self.items)

        self.collection.delete_by_id([1], payload=self.payload)

        # Verify payload was updated
        self.assertIn("collection_name", self.payload)
        self.assertIn("total_vectors", self.payload)
        self.assertEqual(self.payload["collection_name"], "test_collection")

    def test_collection_reindex_with_payload(self):
        """Test Collection.reindex() with non-empty payload."""
        self.collection.insert(self.items)

        self.collection.reindex(payload=self.payload)

        # Verify payload was updated
        self.assertIn("collection_name", self.payload)
        self.assertIn("vector_dim", self.payload)
        self.assertIn("total_vectors", self.payload)
        self.assertEqual(self.payload["collection_name"], "test_collection")

    def test_collection_save_with_payload(self):
        """Test Collection.save() with non-empty payload via Client."""
        self.collection.insert(self.items)

        self.client.save_collection("test_collection", payload=self.payload)

        # Verify payload was updated
        self.assertIn("collection_name", self.payload)
        self.assertIn("vector_dim", self.payload)
        self.assertIn("total_vectors", self.payload)
        self.assertEqual(self.payload["collection_name"], "test_collection")

    def test_collection_load_with_payload(self):
        """Test Collection.load() with non-empty payload via Client."""
        self.collection.insert(self.items)
        self.client.save_collection("test_collection")

        # Create a new client to load the collection
        new_client = Client(self.temp_dir)
        loaded_collection = new_client.get_collection("test_collection")

        # Verify the collection was loaded correctly
        self.assertIsNotNone(loaded_collection)
        result = loaded_collection.filter_query({})
        self.assertEqual(len(result["id"]), 3)

    def test_collection_insert_empty_list(self):
        """Test Collection.insert() with empty list triggers warning."""
        # This should trigger the warning and return early without error
        self.collection.insert([], payload=self.payload)

        # Verify the collection remains empty by checking get_by_id returns empty
        result = self.collection.get_by_id([])
        self.assertEqual(result, {"id": [], "document": [], "metadata": []})

    def test_collection_upsert_empty_list(self):
        """Test Collection.upsert() with empty list triggers warning."""
        # This should trigger the warning and return early without error
        self.collection.upsert([], payload=self.payload)

        # Verify the collection remains empty by checking get_by_id returns empty
        result = self.collection.get_by_id([])
        self.assertEqual(result, {"id": [], "document": [], "metadata": []})

    def test_collection_delete_by_id_empty_list(self):
        """Test Collection.delete_by_id() with empty list triggers warning."""
        # First insert some items
        self.collection.insert(self.items)

        # This should trigger the warning and return early
        self.collection.delete_by_id([], payload=self.payload)

        # All items should still be present
        result = self.collection.filter_query({})
        self.assertEqual(len(result["id"]), 3)


class TestNotifyClient(unittest.TestCase):
    """Test suite for Client class @notify decorated methods with payload."""

    def setUp(self):
        """Set up a new client for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.client = Client(self.temp_dir)
        self.vectors = np.random.rand(50, 64).astype(np.float32)
        self.payload = {"test_key": "test_value"}

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_client_create_collection_with_payload(self):
        """Test Client.create_collection() with non-empty payload."""
        collection = self.client.create_collection("test_col", payload=self.payload)

        # Verify payload was updated
        self.assertIn("collection_name", self.payload)
        self.assertEqual(self.payload["collection_name"], "test_col")
        self.assertIsNotNone(collection)

    def test_client_create_index_with_payload(self):
        """Test Client.create_index() with non-empty payload."""
        index = self.client.create_index("test_idx", payload=self.payload)

        # Verify payload was updated
        self.assertIn("index_type", self.payload)
        self.assertIsNotNone(index)

    def test_client_delete_collection_with_payload(self):
        """Test Client.delete_collection() with non-empty payload."""
        self.client.create_collection("test_col")

        self.client.delete_collection("test_col", delete_on_disk=False, payload=self.payload)

        # Verify payload was updated
        self.assertIn("collection_name", self.payload)
        self.assertEqual(self.payload["collection_name"], "test_col")

    def test_client_delete_index_with_payload(self):
        """Test Client.delete_index() with non-empty payload."""
        index = self.client.create_index("test_idx")
        index.fit(self.vectors)

        self.client.delete_index("test_idx", delete_on_disk=False, payload=self.payload)

        # Verify payload was updated
        self.assertIn("index_type", self.payload)

    def test_client_save_index_with_payload(self):
        """Test Client.save_index() with non-empty payload."""
        index = self.client.create_index("test_idx")
        index.fit(self.vectors)

        self.client.save_index("test_idx", payload=self.payload)

        # Verify payload was updated
        self.assertIn("index_type", self.payload)

    def test_client_save_collection_with_payload(self):
        """Test Client.save_collection() with non-empty payload."""
        collection = self.client.create_collection("test_col")
        items = [
            (1, "Doc 1", [0.1, 0.2, 0.3], {}),
            (2, "Doc 2", [0.4, 0.5, 0.6], {}),
        ]
        collection.insert(items)

        self.client.save_collection("test_col", payload=self.payload)

        # Verify payload was updated
        self.assertIn("collection_name", self.payload)
        self.assertEqual(self.payload["collection_name"], "test_col")


if __name__ == "__main__":
    unittest.main()
