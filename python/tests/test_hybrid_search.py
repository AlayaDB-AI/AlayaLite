# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for hybrid search (vector search + metadata filtering)."""

import os
import shutil
import tempfile
import time
import unittest

import numpy as np
from alayalite import Collection
from alayalite.schema import IndexParams

N_TOTAL = 10000
N_TARGET = 100
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
        """Construct a shared dataset and precompute cosine ground truth."""
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

    def _run_hybrid_search_test(self, collection_name, quant_type, ef_search=300):
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
            query,
            limit=TOP_K,
            metadata_filter={"label": "target_label"},
            ef_search=ef_search,
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

    def test_rabitq_hybrid_search_with_cosine(self):
        """Test hybrid query with RaBitQ quantization using cosine metric."""
        recall, qps = self._run_hybrid_search_test("test_rabitq_cos", "rabitq", ef_search=100)
        print(f"\nRaBitQ + cosine: recall={recall:.4f}, QPS={qps:.2f}")
        self.assertGreaterEqual(recall, 0.95, f"RaBitQ cosine recall too low: {recall:.4f}")

    def test_SQ_hybrid_search_with_cosine(self):
        """Test hybrid search recall with SQ quantization and full precision."""
        for quant_type in ["sq4", "sq8", "none"]:
            with self.subTest(quant=quant_type):
                recall, qps = self._run_hybrid_search_test(f"test_{quant_type}_cos", quant_type, ef_search=150)
                print(f"\n{quant_type.upper()} + cosine: recall={recall:.4f}, QPS={qps:.2f}")
                self.assertGreaterEqual(recall, 0.95, f"{quant_type} cosine recall too low: {recall:.4f}")


if __name__ == "__main__":
    unittest.main()
