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

import platform
import tempfile
import unittest

import numpy as np
from alayalite import Client
from alayalite.index import Index
from alayalite.utils import calc_gt, calc_recall

# Skip RaBitQ tests on non-x86 platforms (AVX512 required)
SKIP_RABITQ = platform.machine() not in ("x86_64", "AMD64")
SKIP_REASON = "RaBitQ requires AVX512 instructions (x86_64 only)"


class TestAlayaLiteRaBitQSearch(unittest.TestCase):
    """Test cases for RaBitQ implementation."""

    def setUp(self):
        self.client = Client()

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
        with tempfile.TemporaryDirectory() as temp_dir:
            self.client = Client(url=temp_dir)

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
            index = Index.load(temp_dir, "rabitq_index")
            result_load = index.batch_search(queries, 10, 400)
            self.assertEqual(result_load.shape, result.shape)
            # result_load equals result
            self.assertTrue(np.allclose(result_load, result))


if __name__ == "__main__":
    unittest.main()
