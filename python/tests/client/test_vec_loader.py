# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for vector loader utilities."""

import json
import os
import tempfile
import unittest

import numpy as np
from alayalite.utils import load_fvecs, load_ivecs, load_npy_jsonl_dataset


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

    def test_load_npy_jsonl_dataset_generates_row_ids(self):
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as npy_file:
            np.save(npy_file, vectors)
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps({"label": "A", "price": 10}) + "\n")
            jsonl_file.write(json.dumps({"label": "B", "price": 20}) + "\n")

        dataset = load_npy_jsonl_dataset(npy_file.name, jsonl_file.name, mmap_mode=None)

        np.testing.assert_array_equal(np.asarray(dataset.vectors), vectors)
        self.assertEqual(dataset.item_ids, ["0", "1"])
        self.assertEqual(dataset.documents, ["", ""])
        self.assertEqual(dataset.metadata_list, [{"label": "A", "price": 10}, {"label": "B", "price": 20}])

        os.unlink(npy_file.name)
        os.unlink(jsonl_file.name)

    def test_load_npy_jsonl_dataset_extracts_id_and_document_fields(self):
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as npy_file:
            np.save(npy_file, vectors)
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps({"item_id": "a", "document": "Doc A", "label": "A"}) + "\n")
            jsonl_file.write(json.dumps({"item_id": "b", "document": "Doc B", "label": "B"}) + "\n")

        dataset = load_npy_jsonl_dataset(
            npy_file.name,
            jsonl_file.name,
            mmap_mode=None,
            item_id_field="item_id",
            document_field="document",
        )

        self.assertEqual(dataset.item_ids, ["a", "b"])
        self.assertEqual(dataset.documents, ["Doc A", "Doc B"])
        self.assertEqual(dataset.metadata_list, [{"label": "A"}, {"label": "B"}])

        os.unlink(npy_file.name)
        os.unlink(jsonl_file.name)

    def test_load_npy_jsonl_dataset_can_keep_only_selected_metadata_fields(self):
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as npy_file:
            np.save(npy_file, vectors)
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps({"item_id": "a", "score": 10, "group": "x"}) + "\n")
            jsonl_file.write(json.dumps({"item_id": "b", "score": 20, "group": "y"}) + "\n")

        dataset = load_npy_jsonl_dataset(
            npy_file.name,
            jsonl_file.name,
            mmap_mode=None,
            item_id_field="item_id",
            metadata_fields=["score"],
        )

        self.assertEqual(dataset.item_ids, ["a", "b"])
        self.assertEqual(dataset.metadata_list, [{"score": 10}, {"score": 20}])

        os.unlink(npy_file.name)
        os.unlink(jsonl_file.name)

    def test_load_npy_jsonl_dataset_skips_blank_lines_without_shifting_row_ids(self):
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as npy_file:
            np.save(npy_file, vectors)
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as jsonl_file:
            jsonl_file.write("\n")
            jsonl_file.write(json.dumps({"score": 10}) + "\n")
            jsonl_file.write("\n")
            jsonl_file.write(json.dumps({"score": 20}) + "\n")

        dataset = load_npy_jsonl_dataset(npy_file.name, jsonl_file.name, mmap_mode=None)

        self.assertEqual(dataset.item_ids, ["0", "1"])
        self.assertEqual(dataset.metadata_list, [{"score": 10}, {"score": 20}])

        os.unlink(npy_file.name)
        os.unlink(jsonl_file.name)


if __name__ == "__main__":
    unittest.main()
