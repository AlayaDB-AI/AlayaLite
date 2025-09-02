import hashlib
import unittest

import numpy as np
from alayalite.utils import md5


class TestMD5Function(unittest.TestCase):
    def test_same_array_same_hash(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        hash1 = md5(arr)
        hash2 = md5(arr)
        self.assertEqual(hash1, hash2)

    def test_different_array_different_hash(self):
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([4, 5, 6], dtype=np.int32)
        hash1 = md5(arr1)
        hash2 = md5(arr2)
        self.assertNotEqual(hash1, hash2)

    def test_large_array(self):
        arr = np.random.rand(10_000_000).astype(np.float32)  # 10 million float32
        expected_hash = hashlib.md5(arr.tobytes()).hexdigest()
        computed_hash = md5(arr)
        self.assertEqual(computed_hash, expected_hash)

    def test_dtype_independence(self):
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([1, 2, 3], dtype=np.float32)
        hash1 = md5(arr1)
        hash2 = md5(arr2)
        self.assertNotEqual(hash1, hash2)

    def test_chunk_size_effect(self):
        arr = np.random.rand(10000).astype(np.float32)
        hash1 = md5(arr, chunk_size=1024)
        hash2 = md5(arr, chunk_size=1024 * 1024)
        self.assertEqual(hash1, hash2)


if __name__ == "__main__":
    unittest.main()
