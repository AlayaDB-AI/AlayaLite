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
Unit tests for the AlayaLite Index, focusing on handling various vector data types.
"""

import unittest

import numpy as np
from alayalite import Client
from alayalite.utils import calc_gt, calc_recall


class TestAlayaLiteIndex(unittest.TestCase):
    """Test suite for index creation, fitting, and searching with different dtypes."""

    def setUp(self):
        self.client = Client()

    def test_float32_vectors(self):
        index = self.client.create_index()
        vectors = np.random.rand(1000, 128).astype(np.float32)
        queries = np.random.rand(10, 128).astype(np.float32)
        index.fit(vectors)
        result = index.batch_search(queries, 10)
        gt = calc_gt(vectors, queries, 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.9)

    def test_single_float32_query(self):
        index = self.client.create_index()
        vectors = np.random.rand(1000, 128).astype(np.float32)
        single_query = np.random.rand(128).astype(np.float32)
        index.fit(vectors)
        result = index.search(single_query, 10).reshape(1, -1)
        gt = calc_gt(vectors, single_query.reshape(1, -1), 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.9)

    def test_int32_vectors(self):
        index = self.client.create_index("IntVectors", data_type=np.int32)
        vectors = np.random.randint(0, 100, (1000, 128), dtype=np.int32)
        queries = np.random.randint(0, 100, (10, 128), dtype=np.int32)
        index.fit(vectors)
        result = index.batch_search(queries, 10)
        gt = calc_gt(vectors, queries, 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.9)

    def test_uint32_vectors(self):
        index = self.client.create_index("UInt32Vectors", data_type=np.uint32)
        vectors = np.random.randint(0, 255, (50, 128), dtype=np.uint32)
        queries = np.random.randint(0, 255, (1, 128), dtype=np.uint32)
        index.fit(vectors)
        result = index.batch_search(queries, 10)
        gt = calc_gt(vectors, queries, 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.9)

    def test_uint8_vectors(self):
        index = self.client.create_index("UInt8Vectors", data_type=np.uint8)
        vectors = np.random.randint(0, 255, (50, 128), dtype=np.uint8)
        queries = np.random.randint(0, 255, (1, 128), dtype=np.uint8)
        index.fit(vectors)
        result = index.batch_search(queries, 10)
        gt = calc_gt(vectors, queries, 10)
        recall = calc_recall(result, gt)
        self.assertGreaterEqual(recall, 0.9)


if __name__ == "__main__":
    unittest.main()
