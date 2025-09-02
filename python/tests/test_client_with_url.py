import os
import tempfile
import unittest

import numpy as np
from alayalite import Client


class TestClientWithURL(unittest.TestCase):
    def setUp(self):
        self.tempDire = tempfile.TemporaryDirectory()
        self.url = self.tempDire.name
        self.client = Client(self.url)
        self.ind_vectors = np.random.rand(1000, 128).astype(np.float32)
        self.coll_items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]), {"category": "B"}),
        ]

    def tearDown(self):
        self.tempDire.cleanup()

    def test_save_and_delete_ind(self):
        with self.assertRaises(RuntimeError):
            self.client.save_index("non-exist")
        index_name = "ind"
        index = self.client.create_index(index_name)
        index.fit(self.ind_vectors)
        self.assertIn(index_name, self.client.list_indices())

        self.client.save_index(index_name)
        index_path = os.path.join(self.url, index_name)
        self.assertTrue(os.path.isdir(index_path))
        schema_path = os.path.join(index_path, "schema.json")
        self.assertTrue(os.path.isfile(schema_path))

        self.client.delete_index(index_name, True)
        self.assertNotIn(index_name, self.client.list_indices())
        self.assertFalse(os.path.isdir(index_path))
        self.assertFalse(os.path.isfile(schema_path))

    def test_save_and_delete_coll(self):
        with self.assertRaises(RuntimeError):
            self.client.save_collection("non-exist")
        coll_name = "coll"
        coll = self.client.create_collection(coll_name)
        coll.insert(self.coll_items)
        self.assertIn(coll_name, self.client.list_collections())

        self.client.save_collection(coll_name)
        coll_path = os.path.join(self.url, coll_name)
        self.assertTrue(os.path.isdir(coll_path))
        schema_path = os.path.join(coll_path, "schema.json")
        self.assertTrue(os.path.isfile(schema_path))

        self.client.delete_collection(coll_name, True)
        self.assertNotIn(coll_name, self.client.list_collections())
        self.assertFalse(os.path.isdir(coll_path))
        self.assertFalse(os.path.isfile(schema_path))

    def test_init_load(self):
        index_name = "ind"
        index = self.client.create_index(
            index_name, metric="ip", quantization_type="sq8", index_type="nsg", max_nbrs=100
        )
        index.fit(self.ind_vectors)
        self.client.save_index(index_name)
        coll_name = "coll"
        coll = self.client.create_collection(coll_name)
        coll.insert(self.coll_items)
        self.client.save_collection(coll_name)
        _ = Client(self.url)  # load

    def test_load_different(self):
        index_name = "ind1"
        index1 = self.client.create_index(
            index_name, metric="cos", quantization_type="sq4", index_type="fusion", id_type=np.uint32
        )
        index1.fit(self.ind_vectors)
        with self.assertRaises(RuntimeError):  # double fit
            index1.fit(self.ind_vectors)
        self.client.save_index(index_name)
        with self.assertRaises(ValueError):
            _ = self.client.create_index("illegal_nbr_ind", max_nbrs=1000)
        _ = Client(self.url)  # load

    def test_data_type_mismatch(self):
        index_name = "ind"
        index = self.client.create_index(index_name, data_type=np.int8)
        with self.assertRaises(ValueError):
            index.fit(self.ind_vectors)


if __name__ == "__main__":
    unittest.main()
