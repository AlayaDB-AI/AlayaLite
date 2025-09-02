import os
import tempfile
import unittest

import numpy as np
from alayalite.utils import load_fvecs, load_ivecs


class TestVectorLoaders(unittest.TestCase):
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
