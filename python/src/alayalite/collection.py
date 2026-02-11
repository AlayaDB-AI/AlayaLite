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
This module defines the Collection class, which manages documents,
their embeddings, and the associated vector index.

Refactored: Data is now stored in C++ Space layer instead of Python DataFrame.
"""

import os
import shutil
from typing import List, Optional

import numpy as np

from ._alayalitepy import LogicOp as _LogicOp
from ._alayalitepy import MetadataFilter as _MetadataFilter
from ._alayalitepy import PyIndexInterface as _PyIndexInterface
from .common import _assert
from .index import Index
from .schema import IndexParams, load_schema


# pylint: disable=unused-private-member
class Collection:
    """
    Collection class to manage a collection of documents and their embeddings.

    Data storage is handled by the underlying C++ Index, supporting:
    - Vectors: stored in the vector space
    - Scalar data (item_id, document, metadata): stored in RocksDB via the Space layer
    """

    def __init__(self, name: str, index_params: IndexParams = None):
        """
        Initializes the collection.

        Args:
            name (str): The name of the collection.
            index_params (IndexParams): Configuration parameters for the index.
        """
        self.__name = name
        self.__index_params = index_params if index_params is not None else IndexParams()
        self.__index_py: Optional[Index] = None
        self.__cpp_index: Optional[_PyIndexInterface] = None

    def _get_cpp_index(self) -> _PyIndexInterface:
        """Get the C++ index, raising error if not initialized."""
        if self.__cpp_index is None:
            raise RuntimeError("Index is not initialized yet")
        return self.__cpp_index

    def batch_query(
        self,
        vectors: List[List[float]],
        limit: int,
        ef_search: int = 100,
        num_threads: int = 1,
    ) -> dict:
        """
        Queries the index using a batch of vectors.

        Returns:
            dict with keys: id, document, metadata, distance
        """
        _assert(self.__index_py is not None, "Index is not initialized yet")
        _assert(len(vectors) > 0, "vectors must not be empty")
        _assert(
            len(vectors[0]) == self.__index_py.get_dim(),
            "Vector dimension must match the index dimension.",
        )
        _assert(num_threads > 0, "num_threads must be greater than 0")
        _assert(ef_search >= limit, "ef_search must be greater than or equal to limit")

        vectors_arr = np.array(vectors, dtype=np.float32)

        # Use Index.batch_search_with_distance (handles cosine normalization internally)
        ids_arr, dists_arr = self.__index_py.batch_search_with_distance(
            vectors_arr,
            limit,
            ef_search,
            num_threads,
        )

        cpp_index = self._get_cpp_index()
        ret = {"id": [], "document": [], "metadata": [], "distance": []}
        for ids_row, dists_row in zip(ids_arr, dists_arr):
            # Batch get item_ids via MultiGet (instead of per-item RocksDB Get)
            item_ids = cpp_index.batch_get_item_ids_by_internal_ids(np.array(ids_row, dtype=np.uint32))
            row_ids = list(item_ids)
            row_dists = [float(d) for d in dists_row]
            # documents and metadata not fetched in batch mode for performance
            row_docs = [""] * len(row_ids)
            row_metas = [{}] * len(row_ids)
            ret["id"].append(row_ids)
            ret["document"].append(row_docs)
            ret["metadata"].append(row_metas)
            ret["distance"].append(row_dists)

        return ret

    def hybrid_query(
        self,
        vectors: List[List[float]],
        limit: int,
        *,
        metadata_filter: Optional[dict] = None,
        ef_search: int = 100,
        num_threads: int = 1,
        bf: bool = False,
        reseed_time: int = 3,
    ) -> dict:
        """
        Queries the index using vectors with metadata filtering.
        """
        _assert(self.__index_py is not None, "Index is not initialized yet")
        _assert(len(vectors) > 0, "vectors must not be empty")
        _assert(ef_search >= limit, "ef_search must be >= limit")

        cpp_index = self._get_cpp_index()
        filter_obj = self._build_filter(metadata_filter)

        vectors_arr = np.array(vectors, dtype=np.float32)
        # Normalize vectors for cosine metric
        if self.__index_params.metric in ("cos", "cosine"):
            norms = np.linalg.norm(vectors_arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            vectors_arr = vectors_arr / norms

        # Returns (ids_array, item_ids_list_of_lists)
        _, item_id_lists = cpp_index.batch_hybrid_search(
            vectors_arr,
            limit,
            ef_search,
            filter_obj,
            num_threads,
            bf,
            reseed_time,
        )

        # Return item_ids directly without fetching scalar data
        return {"id": item_id_lists}

    def filter_query(self, metadata_filter: dict, limit: int = 100) -> dict:
        """
        Filters records based on metadata conditions (without vector search).

        Args:
            metadata_filter: Filter conditions dict, e.g.:
                {"category": "tech"}  # simple equality
                {"score": {"$gt": 80}}  # comparison operator
                {"$and": [{"a": 1}, {"b": 2}]}  # logical combination
            limit: Maximum number of results to return

        Returns:
            dict with keys: id, document, metadata, internal_id
        """
        _assert(self.__index_py is not None, "Index is not initialized yet")
        _assert(limit > 0, "limit must be greater than 0")

        cpp_index = self._get_cpp_index()
        filter_obj = self._build_filter(metadata_filter)

        ids, scalar_list = cpp_index.filter_query(filter_obj, limit)

        return {
            "id": [s.get("item_id", "") for s in scalar_list],
            "document": [s.get("document", "") for s in scalar_list],
            "metadata": [s.get("metadata", {}) for s in scalar_list],
            "internal_id": list(ids),
        }

    def insert(self, items: List[tuple]):
        """
        Inserts multiple documents and their embeddings into the collection.

        Args:
            items: List of tuples (item_id, document, embedding, metadata)
        """
        if not items:
            return

        if self.__index_py is None:
            # First insert - initialize index with batch fit
            _, _, first_embedding, _ = items[0]
            dt = np.array(first_embedding).dtype
            self.__index_params.data_type = dt

            # Check quantization type - Collection requires scalar data support
            self.__index_params.fill_none_values()
            if self.__index_params.quantization_type == "none":
                # Collection requires scalar data, use sq4 as default
                self.__index_params.quantization_type = "sq4"

            # Collection always requires scalar data storage
            self.__index_params.has_scalar_data = True

            # Set RocksDB path based on collection name for isolated storage
            if not self.__index_params.rocksdb_path:
                rocksdb_base = os.environ.get("ALAYALITE_ROCKSDB_DIR", "./RocksDB")
                self.__index_params.rocksdb_path = f"{rocksdb_base}/{self.__name}"

            self.__index_py = Index(self.__name, self.__index_params)

            # Prepare batch data
            vectors = np.array([item[2] for item in items], dtype=dt)
            item_ids = [item[0] for item in items]
            documents = [item[1] for item in items]
            metadata_list = [item[3] for item in items]

            # Fit with scalar data
            self.__index_py.fit(
                vectors,
                ef_construction=400,
                num_threads=1,
                item_ids=item_ids,
                documents=documents,
                metadata_list=metadata_list,
            )
            self.__cpp_index = self.__index_py.get_cpp_index()
        else:
            # Incremental insert with scalar data
            cpp_index = self._get_cpp_index()
            for item_id, document, embedding, metadata in items:
                vec = np.array(embedding, dtype=self.__index_py.get_dtype())
                # Normalize vector for cosine metric
                if self.__index_params.metric in ("cos", "cosine"):
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                cpp_index.insert(
                    vec,
                    100,  # ef
                    item_id,
                    document,
                    metadata or {},
                )

    def upsert(self, items: List[tuple]):
        """
        Inserts new items or updates existing ones.
        """
        if not items:
            return

        if self.__index_py is None:
            self.insert(items)
            return

        cpp_index = self._get_cpp_index()
        new_items_to_insert = []

        for item_id, document, embedding, metadata in items:
            if cpp_index.contains(item_id):
                # Update: remove old, insert new
                cpp_index.remove_by_item_id(item_id)
                vec = np.array(embedding, dtype=self.__index_py.get_dtype())
                # Normalize vector for cosine metric
                if self.__index_params.metric in ("cos", "cosine"):
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                cpp_index.insert(
                    vec,
                    100,  # ef
                    item_id,
                    document,
                    metadata or {},
                )
            else:
                new_items_to_insert.append((item_id, document, embedding, metadata))

        if new_items_to_insert:
            self.insert(new_items_to_insert)

    def delete_by_id(self, ids: List[str]):
        """
        Deletes documents from the collection by their item IDs.
        """
        if not ids or self.__cpp_index is None:
            return

        for item_id in ids:
            try:
                self.__cpp_index.remove_by_item_id(item_id)
            except RuntimeError:
                pass  # item_id not found, skip

    def get_by_id(self, ids: List[str]) -> dict:
        """
        Gets documents from the collection by their item IDs.
        """
        results = {"id": [], "document": [], "metadata": []}

        if not ids or self.__cpp_index is None:
            return results

        for item_id in ids:
            try:
                scalar = self.__cpp_index.get_scalar_data_by_item_id(item_id)
                results["id"].append(scalar.get("item_id", ""))
                results["document"].append(scalar.get("document", ""))
                results["metadata"].append(scalar.get("metadata", {}))
            except RuntimeError:
                pass  # item_id not found, skip

        return results

    def delete_by_filter(self, metadata_filter: dict, batch_size: int = 1000) -> int:
        """
        Deletes items from the collection based on a metadata filter.

        Args:
            metadata_filter: Filter conditions dict, e.g.:
                {"category": "tech"}  # simple equality
                {"score": {"$lt": 50}}  # comparison operator
                {"$or": [{"status": "expired"}, {"status": "deleted"}]}
            batch_size: Number of items to fetch and delete per batch

        Returns:
            Number of items deleted
        """
        _assert(self.__index_py is not None, "Index is not initialized yet")

        total_deleted = 0
        batch_count = batch_size

        while batch_count == batch_size:
            results = self.filter_query(metadata_filter, limit=batch_size)
            item_ids = results.get("id", [])
            batch_count = len(item_ids)
            self.delete_by_id(item_ids)
            total_deleted += batch_count

        return total_deleted

    def reindex(self, ef_construction: int = 400, num_threads: int = 1):
        """
        Rebuilds the index while preserving all data.

        This method extracts all vectors and scalar data from the current index,
        then rebuilds the graph structure with new construction parameters.

        Args:
            ef_construction: Construction parameter for HNSW algorithm
            num_threads: Number of threads for index building
        """
        _assert(self.__index_py is not None, "Index is not initialized yet")
        assert self.__index_py is not None  # for type checker

        cpp_index = self._get_cpp_index()
        data_num = cpp_index.get_data_num()
        dtype = self.__index_py.get_dtype()

        if data_num == 0:
            return

        # Collect all vectors and scalar data
        vectors = []
        item_ids = []
        documents = []
        metadata_list = []

        for i in range(data_num):
            try:
                scalar = cpp_index.get_scalar_data_by_internal_id(i)
                item_id = scalar.get("item_id", "")
                # Skip deleted entries (empty item_id means deleted)
                if not item_id:
                    continue

                vec = cpp_index.get_data_by_id(i)
                vectors.append(vec)
                item_ids.append(item_id)
                documents.append(scalar.get("document", ""))
                metadata_list.append(scalar.get("metadata", {}))
            except RuntimeError:
                # Skip deleted or invalid entries
                continue

        if not vectors:
            return

        # Convert to numpy array
        vectors = np.array(vectors, dtype=dtype)

        # Close old RocksDB connection and remove directory before creating new index
        self.close()

        # Remove old RocksDB directory to allow recreating
        if self.__index_params.rocksdb_path and os.path.exists(self.__index_params.rocksdb_path):
            shutil.rmtree(self.__index_params.rocksdb_path)

        # Create new index with same parameters
        self.__index_py = Index(self.__name, self.__index_params)
        self.__index_py.fit(
            vectors,
            ef_construction=ef_construction,
            num_threads=num_threads,
            item_ids=item_ids,
            documents=documents,
            metadata_list=metadata_list,
        )
        self.__cpp_index = self.__index_py.get_cpp_index()

    def _build_filter(self, filter_dict: Optional[dict]) -> _MetadataFilter:
        """
        Convert Python dict to C++ MetadataFilter.
        """
        mf = _MetadataFilter()
        if filter_dict is None:
            return mf

        for key, value in filter_dict.items():
            if key == "$and":
                for sub_dict in value:
                    sub_filter = self._build_filter(sub_dict)
                    mf.add_sub_filter(sub_filter)
            elif key == "$or":
                mf.logic_op = _LogicOp.OR
                for sub_dict in value:
                    sub_filter = self._build_filter(sub_dict)
                    mf.add_sub_filter(sub_filter)
            elif isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "$eq":
                        mf.add_eq(key, op_value)
                    elif op == "$gt":
                        mf.add_gt(key, op_value)
                    elif op == "$lt":
                        mf.add_lt(key, op_value)
                    elif op == "$in":
                        mf.add_in(key, op_value)
                    else:
                        raise ValueError(f"Unsupported operator: {op}")
            else:
                mf.add_eq(key, value)

        return mf

    def save(self, url):
        """
        Saves the collection to disk.
        """
        if not os.path.exists(url):
            os.makedirs(url)

        schema_map = self.__index_py.save(url)
        schema_map["type"] = "collection"
        return schema_map

    @classmethod
    def load(cls, url, name):
        """
        Loads a collection from disk.
        """
        collection_url = os.path.join(url, name)
        if not os.path.exists(collection_url):
            raise RuntimeError(f"Collection {name} does not exist")

        schema_url = os.path.join(collection_url, "schema.json")
        schema_map = load_schema(schema_url)

        if schema_map.get("type") != "collection":
            raise RuntimeError(f"{name} is not a collection")

        # Restore index params from schema (needed by reindex(), etc.)
        index_params = IndexParams.from_str_dict(schema_map["index"])
        if not index_params.rocksdb_path:
            rocksdb_base = os.environ.get("ALAYALITE_ROCKSDB_DIR", "./RocksDB")
            index_params.rocksdb_path = f"{rocksdb_base}/{name}"

        instance = cls(name, index_params)
        instance.__index_py = Index.load(url, name)
        instance.__cpp_index = instance.__index_py.get_cpp_index()
        return instance

    def set_metric(self, metric: str):
        """
        Sets the metric for the collection's index.
        """
        if self.__index_py is not None:
            raise RuntimeError("Cannot change metric after index is created")

        self.__index_params.metric = metric

    def get_index_params(self):
        """
        Retrieve the configuration parameters of the index in the collection.
        """
        return self.__index_params

    def get_index(self) -> Optional[Index]:
        """
        Get the underlying Index instance.
        """
        return self.__index_py

    def close(self):
        """
        Explicitly close and release RocksDB resources.
        """
        if self.__cpp_index is not None:
            self.__cpp_index.close_db()
            self.__cpp_index = None
        self.__index_py = None

    def __del__(self):
        """
        Destructor
        """
        try:
            self.close()
        except (RuntimeError, AttributeError):
            pass
