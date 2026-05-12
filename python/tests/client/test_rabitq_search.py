# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Test cases for RaBitQ search functionality."""

import os
import shutil
import tempfile
import unittest

import numpy as np
from alayalite import Client
from alayalite.index import Index
from alayalite.utils import calc_gt, calc_recall

RABITQ_TEST_SEED = 12345


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
        self.rng = np.random.default_rng(RABITQ_TEST_SEED)

    def random_vectors(self, shape):
        return self.rng.random(shape).astype(np.float32)

    def tearDown(self):
        """Clean up temp directories after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_rabitq_search_solo(self):
        index = self.client.create_index(name="rabitq_index", metric="l2", quantization_type="rabitq")
        vectors = self.random_vectors((1000, 128))
        single_query = self.random_vectors(128)
        index.fit(vectors)
        result = index.search(single_query, 10, 400).reshape(1, -1)
        gt = calc_gt(vectors, single_query.reshape(1, -1), 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.95)

    def test_rabitq_batch_search(self):
        index = self.client.create_index(name="rabitq_index", metric="l2", quantization_type="rabitq")
        vectors = self.random_vectors((1000, 128))
        queries = self.random_vectors((10, 128))
        index.fit(vectors)
        result = index.batch_search(queries, 10, 400)
        gt = calc_gt(vectors, queries, 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.95)

    def test_rabitq_batch_search_with_distance_unsupported(self):
        index = self.client.create_index(name="rabitq_index", metric="l2", quantization_type="rabitq")
        vectors = self.random_vectors((256, 128))
        queries = self.random_vectors((4, 128))
        index.fit(vectors)

        with self.assertRaises(RuntimeError):
            index.batch_search_with_distance(queries, 10, 400)

    def test_rabitq_save_load(self):
        index = self.client.create_index(name="rabitq_index", metric="l2", quantization_type="rabitq")
        vectors = self.random_vectors((1000, 128))
        queries = self.random_vectors((10, 128))
        index.fit(vectors)
        result = index.batch_search(queries, 10, 400)
        gt = calc_gt(vectors, queries, 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.95)

        self.client.save_index("rabitq_index")
        index = Index.load(self.temp_dir, "rabitq_index")
        result_load = index.batch_search(queries, 10, 400)
        self.assertEqual(result_load.shape, result.shape)
        # result_load equals result
        self.assertTrue(np.allclose(result_load, result))

    def test_rabitq_search_solo_ip(self):
        index = self.client.create_index(name="rabitq_ip_index", metric="ip", quantization_type="rabitq")
        vectors = self.random_vectors((1000, 128))
        single_query = self.random_vectors(128)
        index.fit(vectors)
        result = index.search(single_query, 10, 400).reshape(1, -1)
        gt = calc_gt_ip(vectors, single_query.reshape(1, -1), 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.95)

    def test_rabitq_batch_search_ip(self):
        index = self.client.create_index(name="rabitq_ip_index", metric="ip", quantization_type="rabitq")
        vectors = self.random_vectors((1000, 128))
        queries = self.random_vectors((10, 128))
        index.fit(vectors)
        result = index.batch_search(queries, 10, 400)
        gt = calc_gt_ip(vectors, queries, 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.95)

    def test_rabitq_save_load_ip(self):
        index = self.client.create_index(name="rabitq_ip_index", metric="ip", quantization_type="rabitq")
        vectors = self.random_vectors((1000, 128))
        queries = self.random_vectors((10, 128))
        index.fit(vectors)
        result = index.batch_search(queries, 10, 400)
        gt = calc_gt_ip(vectors, queries, 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.95)

        self.client.save_index("rabitq_ip_index")
        index = Index.load(self.temp_dir, "rabitq_ip_index")
        result_load = index.batch_search(queries, 10, 400)
        self.assertEqual(result_load.shape, result.shape)
        self.assertTrue(np.allclose(result_load, result))

    def test_invalid_parameters(self):
        """Test that ef < k raises an error."""
        index = self.client.create_index(name="rabitq_index", metric="l2", quantization_type="rabitq")
        vectors = self.random_vectors((1000, 128))
        query = self.random_vectors(128)
        index.fit(vectors)

        # ef < k should throw exception
        with self.assertRaises((ValueError, RuntimeError)):
            index.search(query, topk=10, ef_search=5)


if __name__ == "__main__":
    unittest.main()
