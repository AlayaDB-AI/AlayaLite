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

import unittest

import numpy as np

from alayalite import Client


class TestAlayaLiteUpdate(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_insert_vector(self):
        index = self.client.create_index()
        vectors = np.random.rand(1000, 128).astype(np.float32)
        index.fit(vectors)

        new_vector_1 = np.random.rand(128).astype(np.float32)
        id = index.insert(new_vector_1)
        self.assertEqual(id, 1000)

        new_vector_2 = np.random.rand(128).astype(np.float32)
        id = index.insert(new_vector_2)
        self.assertEqual(id, 1001)

        vector = index.get_data_by_id(1000)
        self.assertTrue(np.allclose(vector, new_vector_1))

        vector = index.get_data_by_id(1001)
        self.assertTrue(np.allclose(vector, new_vector_2))

    def test_index_out_of_cope(self):
        index = self.client.create_index(capacity=1000)
        vectors = np.random.rand(1000, 128).astype(np.float32)

        new_vector_1 = np.random.rand(128).astype(np.float32)
        index.fit(vectors)
        self.assertRaises(RuntimeError, index.insert, new_vector_1)


if __name__ == "__main__":
    unittest.main()
