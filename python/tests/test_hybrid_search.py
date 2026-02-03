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

"""Unit tests for hybrid search (vector search + metadata filtering)."""

import os
import platform
import shutil
import tempfile
import unittest

import numpy as np
from alayalite import Collection
from alayalite.schema import IndexParams

# Skip RaBitQ tests on non-x86 platforms (AVX512 required)
SKIP_RABITQ = platform.machine() not in ("x86_64", "AMD64")
SKIP_REASON = "RaBitQ requires AVX512 instructions (x86_64 only)"


class TestHybridSearch(unittest.TestCase):
    """Test suite for hybrid_query with metadata filtering."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["ALAYALITE_ROCKSDB_DIR"] = os.path.join(self.temp_dir, "RocksDB")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_collection(self, name: str, quant_type: str = "sq4") -> Collection:
        params = IndexParams()
        params.quantization_type = quant_type
        return Collection(name, params)

    def _get_basic_collection(self) -> Collection:
        """Create collection with basic 3D test data."""
        collection = self._create_collection("test_hybrid")
        items = [
            (1, "Doc A1", np.array([1.0, 0.0, 0.0]), {"category": "A", "score": 90}),
            (2, "Doc A2", np.array([0.9, 0.1, 0.0]), {"category": "A", "score": 80}),
            (3, "Doc B1", np.array([0.0, 1.0, 0.0]), {"category": "B", "score": 70}),
            (4, "Doc B2", np.array([0.1, 0.9, 0.0]), {"category": "B", "score": 60}),
            (5, "Doc C1", np.array([0.0, 0.0, 1.0]), {"category": "C", "score": 50}),
        ]
        collection.insert(items)
        return collection

    # --- Basic filter operators ---

    def test_simple_eq(self):
        """Test equality filter: {"category": "A"}"""
        collection = self._get_basic_collection()
        result = collection.hybrid_query([[1.0, 0.0, 0.0]], limit=5, metadata_filter={"category": "A"}, ef_search=10)
        for item_id in result["id"][0]:
            if item_id:
                self.assertIn(item_id, ["1", "2"])

    def test_gt_lt(self):
        """Test $gt and $lt operators."""
        collection = self._get_basic_collection()
        # $gt
        result = collection.hybrid_query(
            [[0.5, 0.5, 0.0]], limit=5, metadata_filter={"score": {"$gt": 75}}, ef_search=10
        )
        for item_id in result["id"][0]:
            if item_id:
                self.assertIn(item_id, ["1", "2"])
        # $lt
        result = collection.hybrid_query(
            [[0.0, 0.5, 0.5]], limit=5, metadata_filter={"score": {"$lt": 65}}, ef_search=10
        )
        for item_id in result["id"][0]:
            if item_id:
                self.assertIn(item_id, ["4", "5"])

    def test_in(self):
        """Test $in operator."""
        collection = self._get_basic_collection()
        result = collection.hybrid_query(
            [[0.5, 0.5, 0.0]], limit=5, metadata_filter={"category": {"$in": ["A", "B"]}}, ef_search=10
        )
        for item_id in result["id"][0]:
            if item_id:
                self.assertIn(item_id, ["1", "2", "3", "4"])

    # --- Logical operators ---

    def test_and_or(self):
        """Test $and and $or operators."""
        collection = self._get_basic_collection()
        # $and
        result = collection.hybrid_query(
            [[1.0, 0.0, 0.0]],
            limit=5,
            metadata_filter={"$and": [{"category": "A"}, {"score": {"$gt": 85}}]},
            ef_search=10,
        )
        for item_id in result["id"][0]:
            if item_id:
                self.assertEqual(item_id, "1")
        # $or
        result = collection.hybrid_query(
            [[0.5, 0.5, 0.0]], limit=5, metadata_filter={"$or": [{"category": "A"}, {"category": "C"}]}, ef_search=10
        )
        for item_id in result["id"][0]:
            if item_id:
                self.assertIn(item_id, ["1", "2", "5"])

    def test_nested_and_or(self):
        """Test nested: ($and inside $or) and ($or inside $and)."""
        collection = self._get_basic_collection()
        # (category=A AND score>85) OR (category=B AND score<65)
        result = collection.hybrid_query(
            [[0.5, 0.5, 0.0]],
            limit=5,
            metadata_filter={
                "$or": [
                    {"$and": [{"category": "A"}, {"score": {"$gt": 85}}]},
                    {"$and": [{"category": "B"}, {"score": {"$lt": 65}}]},
                ]
            },
            ef_search=10,
        )
        for item_id in result["id"][0]:
            if item_id:
                self.assertIn(item_id, ["1", "4"])

        # (category IN [A,B]) AND (score > 65 OR score < 55)
        result = collection.hybrid_query(
            [[0.5, 0.5, 0.0]],
            limit=5,
            metadata_filter={
                "$and": [{"category": {"$in": ["A", "B"]}}, {"$or": [{"score": {"$gt": 65}}, {"score": {"$lt": 55}}]}]
            },
            ef_search=10,
        )
        for item_id in result["id"][0]:
            if item_id:
                self.assertIn(item_id, ["1", "2", "3"])

    # --- Edge cases ---

    def test_no_filter(self):
        """Test without filter (should behave like batch_query)."""
        collection = self._get_basic_collection()
        result = collection.hybrid_query([[1.0, 0.0, 0.0]], limit=3, metadata_filter=None, ef_search=10)
        self.assertEqual(len(result["id"]), 1)
        self.assertLessEqual(len(result["id"][0]), 3)

    def test_no_match(self):
        """Test filter that matches nothing."""
        collection = self._get_basic_collection()
        result = collection.hybrid_query([[1.0, 0.0, 0.0]], limit=5, metadata_filter={"category": "D"}, ef_search=10)
        non_empty = [id for id in result["id"][0] if id]
        self.assertEqual(len(non_empty), 0)

    def test_multiple_queries(self):
        """Test with multiple query vectors."""
        collection = self._get_basic_collection()
        result = collection.hybrid_query(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], limit=2, metadata_filter={"score": {"$gt": 55}}, ef_search=10
        )
        self.assertEqual(len(result["id"]), 2)

    # --- Different quantization types ---

    def test_quantization_types(self):
        """Test hybrid query with SQ4 and SQ8 quantization types."""
        for quant_type in ["sq4", "sq8"]:
            with self.subTest(quant=quant_type):
                collection = self._create_collection(f"test_{quant_type}", quant_type)
                items = [
                    (
                        i,
                        f"Doc {i}",
                        np.random.rand(64).astype(np.float32),
                        {"category": "A" if i % 2 else "B", "score": i * 10},
                    )
                    for i in range(1, 6)
                ]
                collection.insert(items)

                query = [list(np.random.rand(64).astype(np.float32))]
                result = collection.hybrid_query(query, limit=3, metadata_filter={"category": "A"}, ef_search=10)
                self.assertEqual(len(result["id"]), 1)
                for item_id in result["id"][0]:
                    if item_id:
                        self.assertIn(item_id, ["1", "3", "5"])

    @unittest.skipIf(SKIP_RABITQ, SKIP_REASON)
    def test_rabitq_hybrid_search(self):
        """Test hybrid query with RaBitQ quantization (requires larger dataset)."""
        collection = self._create_collection("test_rabitq", "rabitq")
        # RaBitQ needs more data
        n_items = 500
        items = [
            (
                i,
                f"Doc {i}",
                np.random.rand(64).astype(np.float32),
                {"category": "A" if i % 2 else "B", "score": i % 100},
            )
            for i in range(1, n_items + 1)
        ]
        collection.insert(items)

        query = [list(np.random.rand(64).astype(np.float32))]
        try:
            result = collection.hybrid_query(query, limit=10, metadata_filter={"category": "A"}, ef_search=50)
        except RuntimeError as _:
            print("AVX512 instruction is not supported.")
            return
        self.assertEqual(len(result["id"]), 1)
        # All returned items should be category A (odd numbers)
        for item_id in result["id"][0]:
            if item_id:
                self.assertEqual(int(item_id) % 2, 1)


if __name__ == "__main__":
    unittest.main()
