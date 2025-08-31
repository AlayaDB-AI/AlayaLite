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
This module provides the Index class, a Python interface for managing
and querying a single vector index.
"""

import os

import numpy as np

from ._alayalitepy import PyIndexInterface as _PyIndexInterface
from .common import (
    VectorLike,
    VectorLikeBatch,
    _assert,
)
from .schema import IndexParams, load_schema


# Pylint is incorrectly flagging used private members.
# pylint: disable=unused-private-member
class Index:
    """
    The Index class provides a Python interface for managing and querying vector indices.
    """

    def __init__(self, name: str = "default", params: IndexParams = IndexParams()):
        """
        Initialize a new Index instance.

        Args:
            name (str): Name identifier for the index. Defaults to "default".
            params (IndexParams): Configuration parameters for the index.
        """
        self.__name = name
        self.__params = params
        self.__index = None  # late initialization
        self.__is_initialized = False
        self.__dim = None  # It will be set when fitting the index

    def get_params(self) -> IndexParams:
        """
        Retrieve the configuration parameters of the index.

        Returns:
            IndexParams: The current parameters used to configure the index.
        """
        return self.__params

    def get_data_by_id(self, vector_id: int) -> VectorLike:
        """
        Retrieve the vector data associated with a given ID.

        Args:
            vector_id (int): The ID of the vector to retrieve.

        Returns:
            VectorLike: The corresponding vector data.
        """
        return self.__index.get_data_by_id(vector_id)

    def fit(
        self,
        vectors: VectorLikeBatch,
        ef_construction: int = 100,
        num_threads: int = 1,
    ):
        """
        Build the index with the given set of vectors.
        """
        if self.__is_initialized:
            raise RuntimeError("An index can be only fitted once")

        _assert(vectors.ndim == 2, "vectors must be a 2D array")
        data_type = np.array(vectors).dtype
        if self.__params.data_type is None:
            self.__params.data_type = data_type
        elif self.__params.data_type != data_type:
            raise ValueError(f"Data type mismatch: {self.__params.data_type} vs {data_type}")

        self.__params.fill_none_values()
        self.__dim = vectors.shape[1]
        self.__index = _PyIndexInterface(self.__params.to_cpp_params())
        self.__is_initialized = True

        print(
            f"fitting index with the following parameters: \n"
            f"  vectors.shape: {vectors.shape}, num_threads: {num_threads}, ef_construction: {ef_construction}\n"
            f"start fitting index..."
        )
        self.__index.fit(vectors, ef_construction, num_threads)

    def insert(self, vectors: VectorLike, ef: int = 100):
        """
        Insert a new vector into the index.
        """
        _assert(self.__index is not None, "Index is not init yet")
        _assert(vectors.ndim == 1, "vectors must be a 1D array")
        _assert(
            vectors.shape[0] == self.__dim,
            f"vectors dimension must match the dimension of the vectors used to fit the index."
            f"fit data dimension: {self.__dim}, vectors dimension: {vectors.shape[0]}",
        )
        ret = self.__index.insert(vectors, ef)
        is_full_uint32 = self.__params.id_type == np.uint32 and ret == 4294967295
        is_full_uint64 = self.__params.id_type == np.uint64 and ret == 18446744073709551615

        if is_full_uint32 or is_full_uint64 or ret == -1:
            raise RuntimeError("The index is full, cannot insert more vectors")
        return ret

    def remove(self, vector_id: int) -> None:
        """
        Remove a vector from the index by ID.
        """
        _assert(self.__index is not None, "Index is not init yet")
        self.__index.remove(vector_id)

    def search(self, query: VectorLike, topk: int, ef_search: int = 100) -> VectorLike:
        """
        Perform a nearest neighbor search for a given query vector.
        """
        _assert(self.__index is not None, "Index is not init yet")
        _assert(query.ndim == 1, "query must be a 1D array")
        _assert(
            query.shape[0] == self.__dim,
            f"query dimension must match the dimension of the vectors used to fit the index."
            f"fit data dimension: {self.__dim}, query dimension: {query.shape[0]}",
        )
        return self.__index.search(query, topk, ef_search)

    def batch_search(
        self,
        queries: VectorLikeBatch,
        topk: int,
        ef_search: int = 100,
        num_threads: int = 1,
    ) -> VectorLikeBatch:
        """
        Perform a batch search for multiple query vectors.
        """
        _assert(self.__index is not None, "Index is not init yet")
        _assert(queries.ndim == 2, "queries must be a 2D array")
        _assert(
            queries.shape[1] == self.__dim,
            f"query dimension must match the dimension of the vectors used to fit the index."
            f"fit data dimension: {self.__dim}, query dimension: {queries.shape[1]}",
        )
        return self.__index.batch_search(queries, topk, ef_search, num_threads)

    def batch_search_with_distance(
        self,
        queries: VectorLikeBatch,
        topk: int,
        ef_search: int = 100,
        num_threads: int = 1,
    ) -> VectorLikeBatch:
        """
        Perform a batch search for multiple query vectors.
        """
        _assert(self.__index is not None, "Index is not init yet")
        _assert(queries.ndim == 2, "queries must be a 2D array")
        _assert(
            queries.shape[1] == self.__dim,
            f"query dimension must match the dimension of the vectors used to fit the index."
            f"fit data dimension: {self.__dim}, query dimension: {queries.shape[1]}",
        )
        return self.__index.batch_search_with_distance(queries, topk, ef_search, num_threads)

    def get_dim(self):
        """
        Get the dimensionality of vectors stored in the index.
        """
        return self.__dim

    def get_dtype(self):
        """
        Get the data type of vectors stored in the index.
        """
        return self.__params.data_type

    def save(self, url) -> dict:
        """
        Save the index to a specified directory.
        """
        if not os.path.exists(url):
            os.makedirs(url)

        index_path = self.__params.index_path(url)
        data_path = self.__params.data_path(url)
        quant_path = self.__params.quant_path(url)

        self.__index.save(index_path, data_path, quant_path)
        return {"type": "index", "index": self.__params.to_json_dict()}

    @classmethod
    def load(cls, url, name):
        """
        Load an existing index from disk.
        """
        index_url = os.path.join(url, name)

        if not os.path.exists(index_url):
            raise RuntimeError("The index file does not exist")

        schema_url = os.path.join(index_url, "schema.json")
        params = IndexParams.from_str_dict(load_schema(schema_url)["index"])
        instance = cls(name, params)
        instance.__index = _PyIndexInterface(params.to_cpp_params())

        index_path = params.index_path(index_url)
        data_path = params.data_path(index_url)
        quant_path = params.quant_path(index_url)

        instance.__index.load(index_path, data_path, quant_path)
        instance.__is_initialized = True
        instance.__dim = instance.__index.get_data_dim()
        return instance
