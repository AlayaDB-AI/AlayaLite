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

"""Test cases for RaBitQ search functionality."""

import os
import platform
import shutil
import tempfile
import unittest

import numpy as np
from alayalite import Client
from alayalite.index import Index
from alayalite.utils import calc_gt, calc_recall

# Skip RaBitQ tests on non-x86 platforms (AVX512 required)
SKIP_RABITQ = platform.machine() not in ("x86_64", "AMD64")
SKIP_REASON = "RaBitQ requires AVX512 instructions (x86_64 only)"


def calc_gt_ip(data, query, topk):
    """Calculate ground truth using inner product (higher is better)."""
    gt = np.zeros((query.shape[0], topk), dtype=np.int32)
    for i in range(query.shape[0]):
        scores = data.astype(np.float64) @ query[i].astype(np.float64)
        gt[i] = np.argsort(-scores)[:topk]
    return gt


class TestAlayaLiteRaBitQSearch(unittest.TestCase):
    """Test cases for RaBitQ implementation."""

    def setUp(self):
        # Create temp directory for test isolation
        self.temp_dir = tempfile.mkdtemp()
        # Set RocksDB directory to temp_dir for isolation
        os.environ["ALAYALITE_ROCKSDB_DIR"] = os.path.join(self.temp_dir, "RocksDB")
        self.client = Client(url=self.temp_dir)

    def tearDown(self):
        """Clean up temp directories after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipIf(SKIP_RABITQ, SKIP_REASON)
    def test_rabitq_search_solo(self):
        index = self.client.create_index(name="rabitq_index", metric="l2", quantization_type="rabitq")
        vectors = np.random.rand(1000, 128).astype(np.float32)
        single_query = np.random.rand(128).astype(np.float32)
        index.fit(vectors)
        try:
            result = index.search(single_query, 10, 400).reshape(1, -1)
        except RuntimeError as _:
            print("AVX512 instruction is not supported.")
            return
        gt = calc_gt(vectors, single_query.reshape(1, -1), 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.95)

    @unittest.skipIf(SKIP_RABITQ, SKIP_REASON)
    def test_rabitq_batch_search(self):
        index = self.client.create_index(name="rabitq_index", metric="l2", quantization_type="rabitq")
        vectors = np.random.rand(1000, 128).astype(np.float32)
        queries = np.random.rand(10, 128).astype(np.float32)
        index.fit(vectors)
        try:
            result = index.batch_search(queries, 10, 400)
        except RuntimeError as _:
            print("AVX512 instruction is not supported.")
            return
        gt = calc_gt(vectors, queries, 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.95)

    @unittest.skipIf(SKIP_RABITQ, SKIP_REASON)
    def test_rabitq_save_load(self):
        index = self.client.create_index(name="rabitq_index", metric="l2", quantization_type="rabitq")
        vectors = np.random.rand(1000, 128).astype(np.float32)
        queries = np.random.rand(10, 128).astype(np.float32)
        index.fit(vectors)
        try:
            result = index.batch_search(queries, 10, 400)
        except RuntimeError as _:
            print("AVX512 instruction is not supported.")
            return
        gt = calc_gt(vectors, queries, 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.95)

        self.client.save_index("rabitq_index")
        index = Index.load(self.temp_dir, "rabitq_index")
        result_load = index.batch_search(queries, 10, 400)
        self.assertEqual(result_load.shape, result.shape)
        # result_load equals result
        self.assertTrue(np.allclose(result_load, result))

    @unittest.skipIf(SKIP_RABITQ, SKIP_REASON)
    def test_rabitq_search_solo_ip(self):
        index = self.client.create_index(name="rabitq_ip_index", metric="ip", quantization_type="rabitq")
        vectors = np.random.rand(1000, 128).astype(np.float32)
        single_query = np.random.rand(128).astype(np.float32)
        index.fit(vectors)
        try:
            result = index.search(single_query, 10, 400).reshape(1, -1)
        except RuntimeError as _:
            print("AVX512 instruction is not supported.")
            return
        gt = calc_gt_ip(vectors, single_query.reshape(1, -1), 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.95)

    @unittest.skipIf(SKIP_RABITQ, SKIP_REASON)
    def test_rabitq_batch_search_ip(self):
        index = self.client.create_index(name="rabitq_ip_index", metric="ip", quantization_type="rabitq")
        vectors = np.random.rand(1000, 128).astype(np.float32)
        queries = np.random.rand(10, 128).astype(np.float32)
        index.fit(vectors)
        try:
            result = index.batch_search(queries, 10, 400)
        except RuntimeError as _:
            print("AVX512 instruction is not supported.")
            return
        gt = calc_gt_ip(vectors, queries, 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.95)

    @unittest.skipIf(SKIP_RABITQ, SKIP_REASON)
    def test_rabitq_save_load_ip(self):
        index = self.client.create_index(name="rabitq_ip_index", metric="ip", quantization_type="rabitq")
        vectors = np.random.rand(1000, 128).astype(np.float32)
        queries = np.random.rand(10, 128).astype(np.float32)
        index.fit(vectors)
        try:
            result = index.batch_search(queries, 10, 400)
        except RuntimeError as _:
            print("AVX512 instruction is not supported.")
            return
        gt = calc_gt_ip(vectors, queries, 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.95)

        self.client.save_index("rabitq_ip_index")
        index = Index.load(self.temp_dir, "rabitq_ip_index")
        result_load = index.batch_search(queries, 10, 400)
        self.assertEqual(result_load.shape, result.shape)
        self.assertTrue(np.allclose(result_load, result))

    @unittest.skipIf(SKIP_RABITQ, SKIP_REASON)
    def test_invalid_parameters(self):
        """Test that ef < k raises an error."""
        index = self.client.create_index(name="rabitq_index", metric="l2", quantization_type="rabitq")
        vectors = np.random.rand(1000, 128).astype(np.float32)
        query = np.random.rand(128).astype(np.float32)
        index.fit(vectors)

        try:
            # ef < k should throw exception
            with self.assertRaises((ValueError, RuntimeError)):
                index.search(query, topk=10, ef_search=5)
        except RuntimeError as _:
            # AVX512 not supported, test passes
            print("AVX512 instruction is not supported.")


if __name__ == "__main__":
    unittest.main()
