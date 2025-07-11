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

import hashlib

import numpy as np

__all__ = ["load_fvecs", "load_ivecs", "calc_recall", "calc_gt"]


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
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for k in range(result.shape[1]):
                if result[i][j] == gt_data[i][k]:
                    cnt += 1
                    break

    return 1.0 * cnt / (len(result) * result.shape[1])


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
