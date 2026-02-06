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

    @unittest.skipIf(SKIP_RABITQ, SKIP_REASON)
    def test_rabitq_hybrid_search(self):
        """Test hybrid query with RaBitQ quantization on 100k dataset with 100 target labels."""
        np.random.seed(42)
        n_total = 1000000
        n_target = 10000
        dim = 64
        top_k = 100

        params = IndexParams()
        params.quantization_type = "rabitq"
        params.capacity = n_total + 1000
        params.indexed_fields = ["label"]

        collection = Collection("test_rabitq", params)

        target_indices = set(np.random.choice(n_total, n_target, replace=False))

        items = []
        for i in range(n_total):
            vec = np.random.rand(dim).astype(np.float32)
            label = "target_label" if i in target_indices else "other"
            items.append((i, f"Doc {i}", vec, {"label": label}))

        collection.insert(items)

        query = [list(np.random.rand(dim).astype(np.float32))]
        result = collection.hybrid_query(query, limit=top_k, metadata_filter={"label": "target_label"}, ef_search=120)
        self.assertEqual(len(result["id"]), 1)

        found_ids = [int(item_id) for item_id in result["id"][0] if item_id]
        # Verify all found items have target_label
        for item_id in found_ids:
            self.assertIn(item_id, target_indices)

        recall = len(found_ids)
        print(f"\nRaBitQ hybrid search (100k dataset): Found {recall}/{top_k} target items")
        self.assertGreaterEqual(recall, top_k, f"RaBitQ recall too low: {recall}/{top_k}")

    def test_100k_recall_with_cosine(self):
        """Test hybrid search recall on 10k dataset with only 100 target labels using cosine metric."""
        np.random.seed(42)
        n_total = 100000
        n_target = 1000
        dim = 128
        top_k = 100

        for quant_type in ["sq4", "sq8"]:
            with self.subTest(quant=quant_type):
                params = IndexParams()
                params.quantization_type = quant_type
                params.metric = "cos"
                params.capacity = n_total + 1000
                params.indexed_fields = ["label"]

                collection = Collection(f"test_10k_recall_{quant_type}", params)

                target_indices = np.random.choice(n_total, n_target, replace=False)
                target_set = set(target_indices)

                items = []
                for i in range(n_total):
                    vec = np.random.rand(dim).astype(np.float32)
                    vec = vec / np.linalg.norm(vec)
                    label = "target_label" if i in target_set else "other"
                    items.append((i, f"Doc {i}", vec, {"label": label}))

                collection.insert(items)

                query_vec = np.random.rand(dim).astype(np.float32)
                query_vec = query_vec / np.linalg.norm(query_vec)

                result = collection.hybrid_query(
                    [query_vec.tolist()],
                    limit=top_k,
                    metadata_filter={"label": "target_label"},
                    ef_search=120,
                )

                found_ids = [int(id) for id in result["id"][0] if id]
                # Verify all found items have target_label
                for item_id in found_ids:
                    self.assertIn(item_id, target_set)

                recall = len(found_ids)
                print(f"\n{quant_type.upper()} + cosine: Found {recall}/{top_k} target items")
                self.assertGreaterEqual(recall, top_k, f"{quant_type} recall too low: {recall}/{top_k}")


if __name__ == "__main__":
    unittest.main()
