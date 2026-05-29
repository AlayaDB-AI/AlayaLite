# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module provides utility functions for vector database operations,
including loading vector files, calculating recall, and generating ground truth data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "ColumnarDataset",
    "load_fvecs",
    "load_ivecs",
    "load_npy_jsonl_dataset",
    "calc_recall",
    "calc_gt",
    "md5",
    "normalize_vectors_for_cosine_metric",
    "normalize_vectors_for_metric",
]


@dataclass
class ColumnarDataset:
    """Columnar batch data that can be fed directly into ``Collection.fit``."""

    vectors: np.ndarray
    item_ids: list[str]
    documents: list[str]
    metadata_list: list[dict]


def load_fvecs(file_path):
    """
    Load fvecs file into numpy array, fvecs file format is:
      <num_of_dimensions> <vector_1>
      <num_of_dimensions> <vector_2>
      ...
      <num_of_dimensions> <vector_n>

    :param file_path: path to the fvecs file
    :return: numpy array of vectors (n x dim)
    """
    vectors = []
    with open(file_path, "rb") as f:
        while True:
            vector = f.read(4)
            if not vector:
                break
            dim = int.from_bytes(vector, byteorder="little")

            vector_bytes = f.read(dim * 4)
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            vectors.append(vector)
    return np.array(vectors)


def load_ivecs(file_path):
    """
    Load ivecs file into numpy array, ivecs file format is:
      <num_of_dimensions> <vector_1>
      <num_of_dimensions> <vector_2>
      ...
      <num_of_dimensions> <vector_n>

    :param file_path: path to the ivecs file
    :return: numpy array of vectors (n x dim)
    """
    vectors = []
    with open(file_path, "rb") as f:
        while True:
            vector = f.read(4)
            if not vector:
                break
            dim = int.from_bytes(vector, byteorder="little")

            vector_bytes = f.read(dim * 4)
            vector = np.frombuffer(vector_bytes, dtype=np.int32)
            vectors.append(vector)

    return np.array(vectors)


def load_npy_jsonl_dataset(
    vectors_path: str,
    payloads_path: str,
    *,
    limit: Optional[int] = None,
    mmap_mode: Optional[str] = "r",
    item_id_field: Optional[str] = None,
    document_field: Optional[str] = None,
    metadata_fields: Optional[list[str]] = None,
) -> ColumnarDataset:
    """
    Load a dense ``.npy`` vector matrix plus line-delimited JSON payloads.

    This is designed for large hybrid-search datasets such as Qdrant's
    ``vectors.npy`` + ``payloads.jsonl`` layout, and avoids constructing
    a huge intermediate ``List[tuple]`` for ``Collection.insert``.

    Args:
        vectors_path: Path to the dense vector ``.npy`` file.
        payloads_path: Path to the JSONL payload file.
        limit: Optional max number of rows to load.
        mmap_mode: ``numpy.load`` mmap mode. Use ``"r"`` by default to reduce peak memory.
        item_id_field: Optional payload field to use as external item ID.
            If omitted, row indices are used.
        document_field: Optional payload field to use as document text.
            If omitted, empty strings are used.
        metadata_fields: Optional subset of payload fields to retain in metadata.
            If omitted, all remaining payload fields are kept.

    Returns:
        ColumnarDataset containing vectors, item IDs, documents, and metadata.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than 0")

    vectors = np.load(vectors_path, mmap_mode=mmap_mode, allow_pickle=False)
    if vectors.ndim != 2:
        raise ValueError("vectors_path must point to a 2D numpy array")
    if limit is not None:
        vectors = vectors[:limit]

    item_ids: list[str] = []
    documents: list[str] = []
    metadata_list: list[dict] = []

    loaded_rows = 0
    with open(payloads_path, encoding="utf-8") as payloads_file:
        for line in payloads_file:
            if limit is not None and loaded_rows >= limit:
                break
            if not line.strip():
                continue

            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("Each JSONL payload row must be an object")

            metadata = dict(payload)
            if item_id_field is None:
                item_id = str(loaded_rows)
            else:
                if item_id_field not in metadata:
                    raise ValueError(f"Missing item_id field: {item_id_field}")
                item_id = str(metadata.pop(item_id_field))

            if document_field is None:
                document = ""
            else:
                if document_field not in metadata:
                    raise ValueError(f"Missing document field: {document_field}")
                raw_document = metadata.pop(document_field)
                document = "" if raw_document is None else str(raw_document)

            if metadata_fields is not None:
                metadata = {field: metadata[field] for field in metadata_fields if field in metadata}

            item_ids.append(item_id)
            documents.append(document)
            metadata_list.append(metadata)
            loaded_rows += 1

    if len(item_ids) != vectors.shape[0]:
        raise ValueError(f"Vector/payload row count mismatch: {vectors.shape[0]} vectors vs {len(item_ids)} payloads")

    return ColumnarDataset(
        vectors=vectors,
        item_ids=item_ids,
        documents=documents,
        metadata_list=metadata_list,
    )


def calc_recall(result, gt_data):
    cnt = 0
    row = result.shape[0]
    col = result.shape[1]
    for i in range(row):
        cnt += len(set(result[i]) & set(gt_data[i]))
    return 1.0 * cnt / (row * col)


def calc_gt(data, query, topk):
    gt = np.zeros((query.shape[0], topk), dtype=np.int32)
    for i in range(query.shape[0]):
        dists = np.linalg.norm(data.astype(np.float64) - query[i].astype(np.float64), axis=1)
        gt[i] = np.argsort(dists)[:topk]

    return gt


def md5(arr, chunk_size=1024 * 1024):
    md5_hash = hashlib.md5()
    arr_bytes = arr.tobytes()
    for i in range(0, len(arr_bytes), chunk_size):
        chunk = arr_bytes[i : i + chunk_size]
        md5_hash.update(chunk)

    return md5_hash.hexdigest()


def normalize_vectors_for_cosine_metric(vectors: np.ndarray, metric: Optional[str]) -> np.ndarray:
    """Normalize vectors only when cosine similarity is configured."""
    if metric not in ("cos", "cosine"):
        return vectors

    if vectors.ndim == 1:
        norms = np.linalg.norm(vectors)
        if norms == 0:
            return vectors
        return vectors / norms

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def normalize_vectors_for_metric(vectors: np.ndarray, metric: Optional[str]) -> np.ndarray:
    """Backward-compatible alias for cosine-only normalization."""
    return normalize_vectors_for_cosine_metric(vectors, metric)
