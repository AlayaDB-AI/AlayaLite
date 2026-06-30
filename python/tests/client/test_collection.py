# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Unit tests for the AlayaLite Collection class.
"""

import gc
import json
import os
import random
import shutil
import tempfile
import unittest
import uuid
from unittest.mock import patch

import numpy as np
from alayalite import Collection
from alayalite.index import Index
from alayalite.schema import IndexParams
from alayalite.utils import calc_gt, calc_recall, normalize_vectors_for_cosine_metric


class TestCollection(unittest.TestCase):
    """Test suite for Collection class operations."""

    def setUp(self):
        """Set up a new collection for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self._collections = []
        self.collection = self._create_collection("test_collection", self._collection_params())

    def tearDown(self):
        """Clean up temp directories after each test."""
        for collection in reversed(self._collections):
            try:
                collection.close()
            except RuntimeError:
                pass
        gc.collect()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _collection_params(self, **kwargs) -> IndexParams:
        rocksdb_path = kwargs.pop("rocksdb_path", os.path.join(self.temp_dir, "RocksDB", uuid.uuid4().hex))
        return IndexParams(rocksdb_path=rocksdb_path, **kwargs)

    def _track_collection(self, collection: Collection) -> Collection:
        self._collections.append(collection)
        return collection

    def _create_collection(self, name: str, params: IndexParams) -> Collection:
        return self._track_collection(Collection(name, params))

    def test_insert(self):
        """Test inserting items into the collection."""
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "B"}),
        ]
        self.collection.insert(items)
        result = self.collection.filter_query({})
        self.assertEqual(len(result["id"]), 2)

    def test_initial_insert_rejects_duplicate_item_id(self):
        """First-batch insert should reject duplicate non-empty item IDs before building."""
        items = [
            ("dup", "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            ("dup", "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "B"}),
        ]

        with self.assertRaisesRegex(RuntimeError, "Duplicate item_id: dup"):
            self.collection.insert(items)

    def test_incremental_insert_rejects_duplicate_item_id_and_keeps_original(self):
        """Incremental insert should not mutate graph/vector/scalar state on duplicate item_id."""
        self.collection.insert([("item_a", "Original", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"})])

        with self.assertRaisesRegex(RuntimeError, "Duplicate item_id: item_a"):
            self.collection.insert(
                [("item_a", "Duplicate", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "B"})]
            )

        result = self.collection.get_by_id(["item_a"])
        self.assertEqual(result["document"], ["Original"])
        self.assertEqual(result["metadata"], [{"category": "A"}])
        self.assertEqual(len(self.collection.filter_query({})["id"]), 1)

    def test_initial_insert_rejects_duplicate_columnar_item_id(self):
        """Collection insert should reject duplicate non-empty item IDs from columnar inputs."""
        vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        item_ids = ["dup", "dup"]
        documents = ["Document 1", "Document 2"]
        metadata_list = [{"category": "A"}, {"category": "B"}]
        items = list(zip(item_ids, documents, vectors, metadata_list))

        with self.assertRaisesRegex(RuntimeError, "Duplicate item_id: dup"):
            self.collection.insert(items)

    def test_initial_insert_failure_leaves_collection_retryable(self):
        bad_items = [
            ("dup", "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            ("dup", "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "B"}),
        ]
        with self.assertRaisesRegex(RuntimeError, "Duplicate item_id: dup"):
            self.collection.insert(bad_items)

        self.collection.insert([("ok", "Document OK", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"})])
        result = self.collection.get_by_id(["ok"])
        self.assertEqual(result["document"], ["Document OK"])

    def test_get_cpp_index_before_first_insert_has_actionable_error(self):
        """Accessing the native index before first insert should explain how to initialize it."""
        with self.assertRaisesRegex(RuntimeError, "Call insert\\(\\) with the first batch of data first"):
            self.collection.get_cpp_index()

    def test_insert_uses_explicit_build_threads(self):
        """First-build fit should honor explicit build thread parameters."""
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "B"}),
        ]
        collection = self._create_collection("test_collection_threads", self._collection_params(build_threads=7))
        with patch.object(Index, "fit", autospec=True, return_value=None) as mock_fit:
            collection.insert(items)

        self.assertEqual(mock_fit.call_args.kwargs["num_threads"], 7)

    def test_upsert_fit_and_concat(self):
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "B"}),
        ]
        self.collection.upsert(items)
        result = self.collection.filter_query({})
        self.assertEqual(len(result["id"]), 2)
        new_items = [
            (3, "Document 3", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            (4, "Document 4", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "B"}),
        ]
        self.collection.upsert(new_items)
        result = self.collection.filter_query({})
        self.assertEqual(len(result["id"]), 4)

    def test_batch_query(self):
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {}),
        ]
        self.collection.insert(items)
        result = self.collection.batch_query([[0.1, 0.2, 0.3]], limit=1, ef_search=10)
        self.assertEqual(len(result["id"]), 1)
        self.assertEqual(result["id"][0][0], "1")  # id is returned as string

    def test_batch_query_multiple_results(self):
        """Test a batch query with multiple queries and expected neighbors."""
        items = [
            (1, "document1", np.array([1.0, 2.0, 3.0], dtype=np.float32), {}),
            (2, "document2", np.array([4.0, 5.0, 6.0], dtype=np.float32), {"category": "B"}),
            (3, "document3", np.array([12.0, 32.0, 31.0], dtype=np.float32), {"category": "C"}),
        ]
        self.collection.insert(items)

        result_single = self.collection.batch_query([[12.0, 32.0, 31.0]], limit=1, ef_search=10)
        self.assertEqual(len(result_single["id"][0]), 1)
        self.assertEqual(int(result_single["id"][0][0]), 3)

        result_multi = self.collection.batch_query([[12.0, 32.0, 31.0], [4.0, 5.0, 6.0]], limit=3, ef_search=3)
        self.assertEqual(len(result_multi["id"][0]), 3)
        self.assertEqual(len(result_multi["id"][1]), 3)
        self.assertEqual(list(map(int, result_multi["id"][0])), [3, 2, 1])
        self.assertEqual(list(map(int, result_multi["id"][1])), [2, 1, 3])

    def test_cpp_batch_get_scalar_data_by_internal_ids(self):
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "B"}),
        ]
        self.collection.insert(items)

        cpp_index = self.collection.get_cpp_index()
        scalars = cpp_index.batch_get_scalar_data_by_internal_ids(np.array([0, 1, 99], dtype=np.uint32))

        self.assertEqual(len(scalars), 3)
        self.assertEqual(scalars[0]["item_id"], "1")
        self.assertEqual(scalars[1]["metadata"]["category"], "B")
        self.assertEqual(scalars[2].get("item_id", ""), "")

    def test_cpp_internal_id_bridge_supports_uint64(self):
        params = self._collection_params(id_type=np.uint64)
        collection = self._create_collection("test_collection_uint64_bridge", params)
        collection.insert([("a", "Document A", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"kind": "seed"})])

        cpp_index = collection.get_cpp_index()
        scalar = cpp_index.get_scalar_data_by_internal_id(np.uint64(0))
        self.assertEqual(scalar["item_id"], "a")

        item_ids = cpp_index.batch_get_item_ids_by_internal_ids(np.array([0], dtype=np.uint64))
        self.assertEqual(list(item_ids), ["a"])

        vector = cpp_index.get_data_by_id(np.uint64(0))
        self.assertEqual(vector.shape[0], 3)

        cpp_index.remove(np.uint64(0))
        self.assertFalse(cpp_index.contains("a"))

    def test_upsert(self):
        """Test updating an existing item in the collection."""
        items = [(1, "Old Doc", np.array([0.1, 0.2, 0.3], dtype=np.float32), {})]
        self.collection.insert(items)
        update_items = [(1, "New Doc", np.array([0.2, 0.3, 0.4], dtype=np.float32), {})]
        self.collection.upsert(update_items)
        result = self.collection.get_by_id([1])
        self.assertEqual(len(result["document"]), 1)
        self.assertEqual(result["document"][0], "New Doc")

    def test_delete_by_id(self):
        """Test deleting an item by its ID."""
        items = [(1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {})]
        self.collection.insert(items)
        self.collection.delete_by_id([1])
        result = self.collection.get_by_id([1])
        self.assertEqual(len(result["id"]), 0)

    def test_get_by_id(self):
        """Test retrieving items by their IDs."""
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "B"}),
        ]
        self.collection.insert(items)
        result = self.collection.get_by_id([1])
        self.assertEqual(len(result["id"]), 1)
        self.assertEqual(result["document"][0], "Document 1")

    def test_delete_by_filter(self):
        """Test deleting items based on a metadata filter."""
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "A"}),
            (3, "Document 3", np.array([0.7, 0.8, 0.9], dtype=np.float32), {"category": "B"}),
        ]
        self.collection.insert(items)
        self.collection.delete_by_filter({"category": "A"})
        result = self.collection.filter_query({})
        self.assertEqual(len(result["id"]), 1)
        self.assertEqual(result["id"][0], "3")

    def test_filter_query(self):
        """Test querying items based on a metadata filter."""
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "B"}),
        ]
        self.collection.insert(items)
        result = self.collection.filter_query({"category": "A"})
        self.assertEqual(len(result["id"]), 1)
        self.assertEqual(result["document"][0], "Document 1")

    def test_hybrid_query_with_none_quantization(self):
        """Collection should preserve quantization=none and still support hybrid search."""
        params = self._collection_params(
            quantization_type="none",
            metric="l2",
            capacity=10,
            indexed_fields=["category"],
        )
        collection = self._create_collection("test_collection_none_quant", params)
        items = [
            ("a", "Document A", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            ("b", "Document B", np.array([0.2, 0.3, 0.4], dtype=np.float32), {"category": "B"}),
            ("c", "Document C", np.array([0.1, 0.2, 0.31], dtype=np.float32), {"category": "A"}),
        ]
        collection.insert(items)

        self.assertEqual(collection.get_index_params().quantization_type, "none")

        filtered = collection.filter_query({"category": "A"})
        self.assertEqual(set(filtered["id"]), {"a", "c"})

        result = collection.hybrid_query(
            [[0.1, 0.2, 0.3]],
            limit=2,
            metadata_filter={"category": "A"},
            ef_search=10,
        )
        self.assertEqual(len(result["id"]), 1)
        self.assertTrue(result["id"][0][0] in {"a", "c"})

    def test_hybrid_query_returns_id_rows_for_batch(self):
        """Hybrid query should return only item-id rows without scalar payload backfilling."""
        params = self._collection_params(
            quantization_type="none",
            metric="l2",
            capacity=10,
            indexed_fields=["category"],
        )
        collection = self._create_collection("test_collection_hybrid_payload", params)
        items = [
            ("a", "Document A", np.array([0.0, 0.0, 1.0], dtype=np.float32), {"category": "A"}),
            ("b", "Document B", np.array([0.0, 1.0, 0.0], dtype=np.float32), {"category": "B"}),
            ("c", "Document C", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"category": "A"}),
        ]
        collection.insert(items)

        result = collection.hybrid_query(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            limit=2,
            metadata_filter={"category": "A"},
            ef_search=10,
            num_threads=2,
        )

        self.assertEqual(len(result["id"]), 2)
        self.assertEqual(result["id"][0][0], "a")
        self.assertIn(result["id"][1][0], {"a", "c"})
        for row_ids in result["id"]:
            self.assertEqual(len(row_ids), 2)

    def test_rabitq_collection_build_with_scalar_data_keeps_space_accessible(self):
        """RaBitQ collection build should succeed with metadata-backed scalar storage."""
        params = self._collection_params(
            quantization_type="rabitq",
            metric="l2",
            capacity=10_000,
            indexed_fields=["category"],
        )
        collection = self._create_collection("test_collection_rabitq_scalar", params)
        total = 10_000
        dim = 64
        ids = np.arange(total, dtype=np.int32)
        vectors = np.zeros((total, dim), dtype=np.float32)
        vectors[ids, ids % dim] = 1.0
        vectors[ids, (ids * 7) % dim] += 0.5
        vectors[ids, (ids * 13) % dim] += 0.25
        labels = np.array(["A", "B", "C"], dtype=object)[ids % 3]
        items = [(str(i), f"Document {i}", vectors[i], {"category": labels[i]}) for i in range(total)]

        try:
            collection.insert(items)
        except (RuntimeError, ValueError) as exc:
            self.skipTest(f"RaBitQ runtime support unavailable in this environment: {exc}")

        cpp_index = collection.get_cpp_index()
        vector = cpp_index.get_data_by_id(0)
        self.assertEqual(vector.shape[0], 64)
        self.assertGreater(float(np.linalg.norm(vector)), 0.0)

        result = collection.filter_query({"category": "A"}, limit=32)
        self.assertEqual(len(result["id"]), 32)
        self.assertTrue(all(meta["category"] == "A" for meta in result["metadata"]))
        self.assertTrue(all(int(item_id) % 3 == 0 for item_id in result["id"]))

    def test_hybrid_query_accepts_iterative_filter_hint(self):
        """Hybrid query should accept the Milvus-style iterative_filter execution hint."""
        params = self._collection_params(
            quantization_type="none",
            metric="l2",
            capacity=10,
            indexed_fields=["category"],
        )
        collection = self._create_collection("test_collection_iterative_hint", params)
        items = [
            ("a", "Document A", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A", "score": 10.0}),
            ("b", "Document B", np.array([0.2, 0.3, 0.4], dtype=np.float32), {"category": "B", "score": 20.0}),
            ("c", "Document C", np.array([0.1, 0.2, 0.31], dtype=np.float32), {"category": "A", "score": 30.0}),
        ]
        collection.insert(items)

        result = collection.hybrid_query(
            [[0.1, 0.2, 0.3]],
            limit=2,
            metadata_filter={"score": {"$gt": 15.0}},
            ef_search=10,
            filter_execution_hint="iterative_filter",
        )

        self.assertEqual(len(result["id"]), 1)
        self.assertEqual(result["id"][0][0], "c")
        self.assertEqual(result["id"][0][1], "b")

    def test_batch_hybrid_query_accepts_auto_filter_hint(self):
        """Batch hybrid query should accept explicit auto planning hints."""
        params = self._collection_params(
            quantization_type="none",
            metric="l2",
            capacity=10,
            indexed_fields=["category"],
        )
        collection = self._create_collection("test_collection_auto_hint", params)
        items = [
            ("a", "Document A", np.array([0.0, 0.0, 1.0], dtype=np.float32), {"category": "A"}),
            ("b", "Document B", np.array([0.0, 1.0, 0.0], dtype=np.float32), {"category": "B"}),
            ("c", "Document C", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"category": "A"}),
        ]
        collection.insert(items)

        default_result = collection.hybrid_query(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            limit=2,
            metadata_filter={"category": "A"},
            ef_search=10,
            num_threads=2,
        )
        auto_result = collection.hybrid_query(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            limit=2,
            metadata_filter={"category": "A"},
            ef_search=10,
            num_threads=2,
            filter_execution_hint="auto",
        )

        self.assertEqual(auto_result["id"], default_result["id"])

    def test_hybrid_query_supports_ge_and_le_filters(self):
        """Hybrid query should support inclusive range operators on indexed numeric fields."""
        params = self._collection_params(
            quantization_type="none",
            metric="l2",
            capacity=10,
            indexed_fields=["price"],
        )
        collection = self._create_collection("test_collection_ge_le", params)
        items = [
            ("a", "Document A", np.array([0.0, 0.0, 1.0], dtype=np.float32), {"price": 10}),
            ("b", "Document B", np.array([0.0, 1.0, 0.0], dtype=np.float32), {"price": 20}),
            ("c", "Document C", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"price": 30}),
        ]
        collection.insert(items)

        le_result = collection.hybrid_query(
            [[0.0, 0.0, 1.0]],
            limit=3,
            metadata_filter={"price": {"$le": 20}},
            ef_search=10,
        )
        ge_result = collection.hybrid_query(
            [[1.0, 0.0, 0.0]],
            limit=3,
            metadata_filter={"price": {"$ge": 20}},
            ef_search=10,
        )

        self.assertEqual(le_result["id"][0][:2], ["a", "b"])
        self.assertEqual(ge_result["id"][0][:2], ["c", "b"])

    def test_hybrid_query_accepts_compiled_metadata_filter(self):
        """Hybrid query should allow reusing a precompiled native metadata filter."""
        params = self._collection_params(
            quantization_type="none",
            metric="l2",
            capacity=10,
            indexed_fields=["category"],
        )
        collection = self._create_collection("test_collection_compiled_filter", params)
        items = [
            ("a", "Document A", np.array([0.0, 0.0, 1.0], dtype=np.float32), {"category": "A"}),
            ("b", "Document B", np.array([0.0, 1.0, 0.0], dtype=np.float32), {"category": "B"}),
            ("c", "Document C", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"category": "A"}),
        ]
        collection.insert(items)

        compiled_filter = collection.build_filter({"category": "A"})
        self.assertIs(compiled_filter, collection.build_filter(compiled_filter))

        result = collection.hybrid_query(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            limit=2,
            metadata_filter=compiled_filter,
            ef_search=10,
            num_threads=2,
        )

        self.assertEqual(result["id"][0][0], "a")
        self.assertIn(result["id"][1][0], {"a", "c"})

    def test_hybrid_query_rejects_unknown_filter_execution_hint(self):
        """Hybrid query should fail fast for unknown execution hints."""
        items = [
            ("a", "Document A", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            ("b", "Document B", np.array([0.2, 0.3, 0.4], dtype=np.float32), {"category": "B"}),
        ]
        self.collection.insert(items)

        with self.assertRaisesRegex(ValueError, "filter_execution_hint"):
            self.collection.hybrid_query(
                [[0.1, 0.2, 0.3]],
                limit=1,
                metadata_filter={"category": "A"},
                ef_search=10,
                filter_execution_hint="unsupported_hint",
            )

    def test_hybrid_query_materialized_view_merges_multiple_partitions(self):
        """Eligible IN filters should use MV partitions and keep exact global ordering."""
        params = self._collection_params(
            quantization_type="sq8",
            metric="l2",
            capacity=16,
            indexed_fields=["category"],
        )
        collection = self._create_collection("test_collection_mv_merge", params)
        items = [
            ("a1", "Doc A1", np.array([0.0, 0.0, 0.0], dtype=np.float32), {"category": "A", "score": 0}),
            ("a2", "Doc A2", np.array([0.2, 0.0, 0.0], dtype=np.float32), {"category": "A", "score": 2}),
            ("b1", "Doc B1", np.array([0.4, 0.0, 0.0], dtype=np.float32), {"category": "B", "score": 3}),
            ("c1", "Doc C1", np.array([0.6, 0.0, 0.0], dtype=np.float32), {"category": "C", "score": 4}),
            ("c2", "Doc C2", np.array([0.8, 0.0, 0.0], dtype=np.float32), {"category": "C", "score": 5}),
        ]
        collection.insert(items)

        cpp_index = collection.get_cpp_index()
        self.assertEqual(cpp_index.get_materialized_view_partition_count(), 3)

        result = collection.hybrid_query(
            [[0.55, 0.0, 0.0]],
            limit=3,
            metadata_filter={"category": {"$in": ["A", "C"]}, "score": {"$gt": 1}},
            ef_search=10,
        )

        self.assertEqual(result["id"][0], ["c1", "c2", "a2"])

    def test_hybrid_query_materialized_view_partition_only_eq_filter(self):
        """Pure partition-field EQ filters should return partition-local ANN results correctly."""
        params = self._collection_params(
            quantization_type="sq8",
            metric="l2",
            capacity=16,
            indexed_fields=["category"],
        )
        collection = self._create_collection("test_collection_mv_partition_only", params)
        items = [
            ("a1", "Doc A1", np.array([0.0, 0.0, 0.0], dtype=np.float32), {"category": "A"}),
            ("a2", "Doc A2", np.array([0.2, 0.0, 0.0], dtype=np.float32), {"category": "A"}),
            ("b1", "Doc B1", np.array([0.8, 0.0, 0.0], dtype=np.float32), {"category": "B"}),
            ("b2", "Doc B2", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"category": "B"}),
        ]
        collection.insert(items)

        cpp_index = collection.get_cpp_index()
        self.assertEqual(cpp_index.get_materialized_view_partition_count(), 2)

        result = collection.hybrid_query(
            [[0.1, 0.0, 0.0]],
            limit=2,
            metadata_filter={"category": "A"},
            ef_search=10,
            filter_execution_hint="iterative_filter",
        )

        self.assertEqual(result["id"][0], ["a1", "a2"])

    def test_materialized_view_invalidates_after_incremental_insert(self):
        """Incremental updates should drop stale MV partitions and fall back to runtime hybrid search."""
        params = self._collection_params(
            quantization_type="none",
            metric="l2",
            capacity=16,
            indexed_fields=["category"],
        )
        collection = self._create_collection("test_collection_mv_invalidate", params)
        items = [
            ("a1", "Doc A1", np.array([0.0, 0.0, 0.0], dtype=np.float32), {"category": "A"}),
            ("a2", "Doc A2", np.array([0.2, 0.0, 0.0], dtype=np.float32), {"category": "A"}),
            ("b1", "Doc B1", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"category": "B"}),
        ]
        collection.insert(items)

        cpp_index = collection.get_cpp_index()
        self.assertEqual(cpp_index.get_materialized_view_partition_count(), 2)

        collection.insert([("a3", "Doc A3", np.array([0.05, 0.0, 0.0], dtype=np.float32), {"category": "A"})])

        self.assertEqual(cpp_index.get_materialized_view_partition_count(), 0)

        result = collection.hybrid_query(
            [[0.05, 0.0, 0.0]],
            limit=1,
            metadata_filter={"category": "A"},
            ef_search=10,
        )
        self.assertEqual(result["id"][0][0], "a3")

    def test_rabitq_batch_hybrid_query_uses_materialized_view_without_crashing(self):
        """Batch MV hybrid search should handle RaBitQ partition-local ids larger than 31."""
        params = self._collection_params(
            quantization_type="rabitq",
            metric="l2",
            capacity=600,
            indexed_fields=["labels", "id"],
        )
        collection = self._create_collection("test_collection_rabitq_mv_batch", params)

        items = []
        # Keep the dimensionality above the per-label partition size so each query has a unique
        # nearest neighbor inside the partition; smaller periodic dimensions (for example 64) make
        # ids like 331 and 11 share the exact same embedding and turn this into a flaky tie-break
        # test instead of an ID-mapping regression test.
        dim = 128
        for i in range(500):
            vector = np.zeros(dim, dtype=np.float32)
            vector[i % dim] = 1.0
            vector[(i * 7) % dim] += 0.5
            vector[(i * 13) % dim] += 0.25
            items.append(
                (
                    str(i),
                    f"Doc {i}",
                    vector,
                    {"labels": f"label_{i % 10}", "id": i},
                )
            )
        try:
            collection.insert(items)
        except (RuntimeError, ValueError) as exc:
            self.skipTest(f"RaBitQ runtime support unavailable in this environment: {exc}")

        cpp_index = collection.get_cpp_index()
        self.assertEqual(cpp_index.get_materialized_view_partition_count(), 10)

        target_ids = [331, 341, 351, 361, 371]
        queries = [items[i][2].tolist() for i in target_ids]

        result = collection.hybrid_query(
            queries,
            limit=5,
            metadata_filter={"labels": "label_1"},
            ef_search=50,
            num_threads=1,
        )

        self.assertEqual(len(result["id"]), len(target_ids))
        for row, target_id in zip(result["id"], target_ids):
            self.assertEqual(row[0], str(target_id))
            self.assertTrue(all(int(item_id) % 10 == 1 for item_id in row if item_id))

    def test_mixed_operations(self):
        """Test a sequence of insert, delete, and query operations."""
        uuids = [f"id_{i}" for i in range(1, 1000)]
        documents = [f"Document {i}" for i in range(1, 1000)]
        embeddings = [np.array([random.random() for _ in range(100)], dtype=np.float32) for _ in range(999)]
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

    def test_batch_query_normalizes_cosine_queries_via_common_validation(self):
        params = self._collection_params(metric="cosine", capacity=10)
        collection = self._create_collection("test_collection_cosine_batch", params)
        items = [
            ("a", "Document A", np.array([1.0, 0.0, 0.0], dtype=np.float32), {}),
            ("b", "Document B", np.array([0.0, 1.0, 0.0], dtype=np.float32), {}),
        ]
        collection.insert(items)

        query = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        result = collection.batch_query(query, limit=1, ef_search=10)

        self.assertEqual(result["id"][0][0], "a")
        normalized = normalize_vectors_for_cosine_metric(query, "cosine")
        self.assertAlmostEqual(float(normalized[0][0]), 1.0, places=6)

    def test_load_preserves_explicit_rocksdb_path_from_schema(self):
        client_dir = os.path.join(self.temp_dir, "client")
        os.makedirs(client_dir, exist_ok=True)

        rocksdb_path = os.path.join(client_dir, "custom-rocksdb")
        params = self._collection_params(rocksdb_path=rocksdb_path)
        collection = self._create_collection("persisted", params)
        collection.insert([("a", "Document A", np.array([1.0, 0.0, 0.0], dtype=np.float32), {})])
        save_dir = os.path.join(client_dir, "persisted")
        schema_map = collection.save(save_dir)
        with open(os.path.join(save_dir, "schema.json"), "w", encoding="utf-8") as f:
            json.dump(schema_map, f, indent=4)
        collection.close()

        loaded = self._track_collection(Collection.load(client_dir, "persisted"))

        self.assertEqual(loaded.get_index_params().rocksdb_path, rocksdb_path)

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
        print(
            "collection dtype before insert:",
            self.collection.get_index_params().data_type,
        )
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
        self.assertEqual(set(result_remaining["id"]), {str(i) for i in remaining_ids})

        # Step 3: Reindex to rebuild the graph with only remaining items
        self.collection.reindex()

        # --- Recall check after reindex ---
        remaining_vectors = vectors[900:]  # 100 remaining vectors
        query_indices = np.random.choice(len(remaining_vectors), 20, replace=True)
        queries_after_reindex = remaining_vectors[query_indices]
        result_after_reindex = self.collection.batch_query(queries_after_reindex.tolist(), limit=10, ef_search=100)
        retrieved_after_reindex = np.array(result_after_reindex["id"], dtype=int)
        gt_after_reindex = calc_gt(remaining_vectors, queries_after_reindex, 10)
        recall_after_reindex = calc_recall(np.array([each - 900 for each in retrieved_after_reindex]), gt_after_reindex)
        print(f"Recall after reindex: {recall_after_reindex:.4f}")

        self.assertGreaterEqual(
            recall_after_reindex,
            0.9,
            f"Recall after reindex too low: {recall_after_reindex:.4f}",
        )
        self.assertGreaterEqual(recall_after_reindex, 0.9, f"Recall too low after reindex: {recall}")


if __name__ == "__main__":
    unittest.main()
