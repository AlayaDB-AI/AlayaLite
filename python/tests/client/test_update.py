# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Unit tests for the update functionality of the AlayaLite Index,
such as inserting vectors and handling capacity limits.
"""

import unittest

import numpy as np
from alayalite import Client


class TestAlayaLiteUpdate(unittest.TestCase):
    """Test suite for index update operations."""

    def setUp(self):
        """Set up a new client for each test."""
        self.client = Client()

    def test_insert_vector(self):
        """Test inserting new vectors into a fitted index."""
        index = self.client.create_index()
        vectors = np.random.rand(1000, 128).astype(np.float32)
        index.fit(vectors)

        new_vector_1 = np.random.rand(128).astype(np.float32)
        new_id_1 = index.insert(new_vector_1)
        self.assertEqual(new_id_1, 1000)

        new_vector_2 = np.random.rand(128).astype(np.float32)
        new_id_2 = index.insert(new_vector_2)
        self.assertEqual(new_id_2, 1001)

        vector_1 = index.get_data_by_id(1000)
        self.assertTrue(np.allclose(vector_1, new_vector_1))

        vector_2 = index.get_data_by_id(1001)
        self.assertTrue(np.allclose(vector_2, new_vector_2))

    def test_index_out_of_scope(self):
        """Test that inserting into a full index raises a RuntimeError."""
        index = self.client.create_index(capacity=1000)
        vectors = np.random.rand(1000, 128).astype(np.float32)
        index.fit(vectors)

        new_vector_1 = np.random.rand(128).astype(np.float32)
        with self.assertRaises(RuntimeError):
            index.insert(new_vector_1)


if __name__ == "__main__":
    unittest.main()
