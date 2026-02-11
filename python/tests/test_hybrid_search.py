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
import time
import unittest

import numpy as np
from alayalite import Collection
from alayalite.schema import IndexParams

# Skip RaBitQ tests on non-x86 platforms (AVX512 required)
SKIP_RABITQ = platform.machine() not in ("x86_64", "AMD64")
SKIP_REASON = "RaBitQ requires AVX512 instructions (x86_64 only)"

N_TOTAL = 1000000
N_TARGET = 10000
DIM = 256
TOP_K = 100


def _calc_cosine_gt(vectors, target_indices, query, topk):
    """Compute ground truth top-k among target vectors using cosine similarity."""
    target_ids = np.array(list(target_indices))
    target_vecs = vectors[target_ids]
    # vectors and query are already L2-normalized, so dot product = cosine similarity
    scores = target_vecs @ query
    top_idx = np.argsort(-scores)[:topk]
    return set(target_ids[top_idx].tolist())


class TestHybridSearch(unittest.TestCase):
    """Test suite for hybrid_query with metadata filtering."""

    @classmethod
    def setUpClass(cls):
        """Construct shared 1M dataset with 10k target labels and precompute cosine GT."""
        np.random.seed(42)

        # Generate and normalize vectors
        cls.vectors = np.random.rand(N_TOTAL, DIM).astype(np.float32)
        norms = np.linalg.norm(cls.vectors, axis=1, keepdims=True)
        cls.vectors = cls.vectors / norms

        # Select target indices
        cls.target_indices = set(np.random.choice(N_TOTAL, N_TARGET, replace=False).tolist())

        # Generate and normalize query
        cls.query_vec = np.random.rand(DIM).astype(np.float32)
        cls.query_vec = cls.query_vec / np.linalg.norm(cls.query_vec)

        # Precompute ground truth
        cls.gt_ids = _calc_cosine_gt(cls.vectors, cls.target_indices, cls.query_vec, TOP_K)

        # Prepare items list (shared across tests)
        cls.items = []
        for i in range(N_TOTAL):
            label = "target_label" if i in cls.target_indices else "other"
            cls.items.append((i, f"Doc {i}", cls.vectors[i], {"label": label}))

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["ALAYALITE_ROCKSDB_DIR"] = os.path.join(self.temp_dir, "RocksDB")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _run_hybrid_search_test(self, collection_name, quant_type, ef_search=300, reseed_time=3):
        """Build index, run hybrid search, return recall and QPS."""
        params = IndexParams()
        params.quantization_type = quant_type
        params.metric = "cos"
        params.capacity = N_TOTAL + 1000
        params.indexed_fields = ["label"]

        collection = Collection(collection_name, params)
        collection.insert(self.items)

        query = [self.query_vec.tolist()]

        start = time.perf_counter()
        result = collection.hybrid_query(
            query, limit=TOP_K, metadata_filter={"label": "target_label"}, ef_search=ef_search, reseed_time=reseed_time
        )
        elapsed = time.perf_counter() - start

        self.assertEqual(len(result["id"]), 1)
        found_ids = {int(item_id) for item_id in result["id"][0] if item_id}

        # All found items must have target_label
        for item_id in found_ids:
            self.assertIn(item_id, self.target_indices)

        recall = len(found_ids & self.gt_ids) / len(self.gt_ids)
        qps = 1.0 / elapsed if elapsed > 0 else float("inf")
        return recall, qps

    @unittest.skipIf(SKIP_RABITQ, SKIP_REASON)
    def test_rabitq_hybrid_search_with_cosine(self):
        """Test hybrid query with RaBitQ quantization using cosine metric."""
        recall, qps = self._run_hybrid_search_test("test_rabitq_cos", "rabitq", ef_search=100, reseed_time=1)
        print(f"\nRaBitQ + cosine: recall={recall:.4f}, QPS={qps:.2f}")
        self.assertGreaterEqual(recall, 0.90, f"RaBitQ cosine recall too low: {recall:.4f}")

    def test_SQ_hybrid_search_with_cosine(self):
        """Test hybrid search recall with SQ4/SQ8 quantization using cosine metric."""
        for quant_type in ["sq4", "sq8"]:
            with self.subTest(quant=quant_type):
                recall, qps = self._run_hybrid_search_test(
                    f"test_{quant_type}_cos", quant_type, ef_search=150, reseed_time=2
                )
                print(f"\n{quant_type.upper()} + cosine: recall={recall:.4f}, QPS={qps:.2f}")
                self.assertGreaterEqual(recall, 0.90, f"{quant_type} cosine recall too low: {recall:.4f}")


# Configuration for 1m dataset with 0.1% target (1000 items)
N_TOTAL_LARGE = 1000000
N_TARGET_LARGE = 100000  # 10% of 1m
DIM_LARGE = 256
TOP_K_LARGE = 100


class TestRaBitQBruteForceLarge(unittest.TestCase):
    """Test suite for RaBitQ hybrid_query with brute-force on 1M dataset (0.1% target)."""

    @classmethod
    def setUpClass(cls):
        """Construct 1M dataset with 1k target labels (0.1%) and precompute cosine GT."""
        np.random.seed(42)

        # Generate and normalize vectors
        cls.vectors = np.random.rand(N_TOTAL_LARGE, DIM_LARGE).astype(np.float32)
        norms = np.linalg.norm(cls.vectors, axis=1, keepdims=True)
        cls.vectors = cls.vectors / norms

        # Select target indices (0.1% = 1000 out of 1M)
        cls.target_indices = set(np.random.choice(N_TOTAL_LARGE, N_TARGET_LARGE, replace=False).tolist())

        # Generate and normalize query
        cls.query_vec = np.random.rand(DIM_LARGE).astype(np.float32)
        cls.query_vec = cls.query_vec / np.linalg.norm(cls.query_vec)

        # Precompute ground truth
        cls.gt_ids = _calc_cosine_gt(cls.vectors, cls.target_indices, cls.query_vec, TOP_K_LARGE)

        # Prepare items list (shared across tests)
        cls.items = []
        for i in range(N_TOTAL_LARGE):
            label = "target_label" if i in cls.target_indices else "other"
            cls.items.append((i, f"Doc {i}", cls.vectors[i], {"label": label}))

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["ALAYALITE_ROCKSDB_DIR"] = os.path.join(self.temp_dir, "RocksDB")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _run_rabitq_bf_test(self, collection_name, use_bf=False, ef_search=300, reseed_time=3):
        """Build RaBitQ index, run hybrid search with/without BF, return recall and QPS."""
        params = IndexParams()
        params.quantization_type = "rabitq"
        params.metric = "cos"
        params.capacity = N_TOTAL_LARGE + 1000
        params.indexed_fields = ["label"]

        collection = Collection(collection_name, params)
        print(f"  Inserting {N_TOTAL_LARGE} items...")
        collection.insert(self.items)

        query = [self.query_vec.tolist()]

        print(f"  Running hybrid_query (bf={use_bf}, ef_search={ef_search}, reseed_time={reseed_time})...")
        start = time.perf_counter()
        result = collection.hybrid_query(
            query,
            limit=TOP_K_LARGE,
            metadata_filter={"label": "target_label"},
            ef_search=ef_search,
            bf=use_bf,
            reseed_time=reseed_time,
        )
        elapsed = time.perf_counter() - start

        self.assertEqual(len(result["id"]), 1)
        found_ids = {int(item_id) for item_id in result["id"][0] if item_id}

        # All found items must have target_label
        for item_id in found_ids:
            self.assertIn(item_id, self.target_indices)

        recall = len(found_ids & self.gt_ids) / len(self.gt_ids) if self.gt_ids else 0.0
        qps = 1.0 / elapsed if elapsed > 0 else float("inf")
        return recall, qps, elapsed

    @unittest.skipIf(SKIP_RABITQ, SKIP_REASON)
    def test_rabitq_1m_001pct_comparison(self):
        """Compare RaBitQ graph-based vs brute-force search on the same index."""
        # Build index once
        params = IndexParams()
        params.quantization_type = "rabitq"
        params.metric = "cos"
        params.capacity = N_TOTAL_LARGE + 1000
        params.indexed_fields = ["label"]

        collection = Collection("test_rabitq_1m_001pct_cmp", params)
        print(f"  Inserting {N_TOTAL_LARGE} items...")
        collection.insert(self.items)

        query = [self.query_vec.tolist()]
        ef_search = 100
        reseed_time = 0

        # Search 1: Graph-based
        print(f"  Running hybrid_query (bf=False, ef_search={ef_search}, reseed_time={reseed_time})...")
        start = time.perf_counter()
        result = collection.hybrid_query(
            query,
            limit=TOP_K_LARGE,
            metadata_filter={"label": "target_label"},
            ef_search=ef_search,
            bf=False,
            reseed_time=reseed_time,
        )
        time_graph = time.perf_counter() - start

        self.assertEqual(len(result["id"]), 1)
        found_ids_graph = {int(item_id) for item_id in result["id"][0] if item_id}
        for item_id in found_ids_graph:
            self.assertIn(item_id, self.target_indices)

        recall_graph = len(found_ids_graph & self.gt_ids) / len(self.gt_ids) if self.gt_ids else 0.0
        qps_graph = 1.0 / time_graph if time_graph > 0 else float("inf")

        # Search 2: Brute-force
        print(f"  Running hybrid_query (bf=True, ef_search={ef_search}, reseed_time={reseed_time})...")
        start = time.perf_counter()
        result = collection.hybrid_query(
            query,
            limit=TOP_K_LARGE,
            metadata_filter={"label": "target_label"},
            bf=True,
        )
        time_bf = time.perf_counter() - start

        self.assertEqual(len(result["id"]), 1)
        found_ids_bf = {int(item_id) for item_id in result["id"][0] if item_id}
        for item_id in found_ids_bf:
            self.assertIn(item_id, self.target_indices)

        recall_bf = len(found_ids_bf & self.gt_ids) / len(self.gt_ids) if self.gt_ids else 0.0
        qps_bf = 1.0 / time_bf if time_bf > 0 else float("inf")

        print("  ==================" + "target: " + str(N_TARGET_LARGE) + "=====================\n")
        print(f"  Graph-based: recall={recall_graph:.4f}, QPS={qps_graph:.2f}, time={time_graph:.4f}s")
        print(f"  Brute-force: recall={recall_bf:.4f}, QPS={qps_bf:.2f}, time={time_bf:.4f}s")
        print(f"  Speedup (time): {time_bf / time_graph:.2f}x")
        print(f"  Recall improvement (BF): {(recall_bf - recall_graph):.4f}")
        print("  ============================================================\n")

        # Both should have high recall
        self.assertGreaterEqual(recall_graph, 0.90)
        self.assertGreaterEqual(recall_bf, 0.95)


if __name__ == "__main__":
    unittest.main()
