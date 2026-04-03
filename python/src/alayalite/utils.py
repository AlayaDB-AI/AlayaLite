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
This module provides utility functions for vector database operations,
including loading vector files, calculating recall, and generating ground truth data.
"""

import hashlib
from typing import Optional

import numpy as np

__all__ = [
    "load_fvecs",
    "load_ivecs",
    "calc_recall",
    "calc_gt",
    "md5",
    "normalize_vectors_for_cosine_metric",
    "normalize_vectors_for_metric",
]


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
