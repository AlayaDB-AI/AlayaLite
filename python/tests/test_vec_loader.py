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

"""Tests for vector loader utilities."""

import os
import tempfile
import unittest

import numpy as np
from alayalite.utils import load_fvecs, load_ivecs


class TestVectorLoaders(unittest.TestCase):
    """Test cases for fvecs and ivecs loader functions."""

    def test_load_fvecs(self):
        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            for vec in expected:
                dim = len(vec)
                tmpfile.write(dim.to_bytes(4, byteorder="little"))
                tmpfile.write(vec.tobytes())

        result = load_fvecs(tmpfile.name)
        np.testing.assert_array_equal(result, expected)
        os.unlink(tmpfile.name)

    def test_load_ivecs(self):
        expected = np.array([[1, 2], [3, 4]], dtype=np.int32)

        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            for vec in expected:
                dim = len(vec)
                tmpfile.write(dim.to_bytes(4, byteorder="little"))
                tmpfile.write(vec.tobytes())

        result = load_ivecs(tmpfile.name)
        np.testing.assert_array_equal(result, expected)
        os.unlink(tmpfile.name)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            pass

        result_f = load_fvecs(tmpfile.name)
        self.assertEqual(result_f.shape, (0,))

        result_i = load_ivecs(tmpfile.name)
        self.assertEqual(result_i.shape, (0,))

        os.unlink(tmpfile.name)


if __name__ == "__main__":
    unittest.main()
