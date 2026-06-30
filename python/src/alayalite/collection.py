# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module defines the Collection class, which manages documents,
their embeddings, and the associated vector index.
"""

import os
import shutil
from typing import List, Optional, Union

import numpy as np

from ._alayalitepy import LogicOp as _LogicOp
from ._alayalitepy import MetadataFilter as _MetadataFilter
from ._alayalitepy import PyIndexInterface as _PyIndexInterface
from .common import _assert, _validate_query_vectors, normalize_filter_execution_hint, valid_dtype
from .index import Index
from .schema import IndexParams, load_schema, save_schema
from .utils import normalize_vectors_for_cosine_metric


def _default_rocksdb_path(collection_name: str, base_dir: Optional[str] = None) -> str:
    if base_dir is not None:
        return os.path.join(base_dir, collection_name, "rocksdb")

    rocksdb_base = os.environ.get("ALAYALITE_ROCKSDB_DIR", "./RocksDB")
    return os.path.join(rocksdb_base, collection_name)


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
        if not self.__index_params.rocksdb_path:
            self.__index_params.rocksdb_path = _default_rocksdb_path(name)
        self.__index_py: Optional[Index] = None
        self.__cpp_index: Optional[_PyIndexInterface] = None

    def get_cpp_index(self) -> _PyIndexInterface:
        """
        Get the underlying native index.
        """
        return self._get_cpp_index()

    def build_filter(self, filter_dict: Optional[Union[dict, _MetadataFilter]]) -> _MetadataFilter:
        """
        Compile a Python metadata filter into the native representation.
        """
        return self._build_filter(filter_dict)

    def _get_cpp_index(self) -> _PyIndexInterface:
        """Get the C++ index, raising error if not initialized."""
        if self.__index_py is None:
            raise RuntimeError(
                "Collection index is not initialized yet. Call insert() with the first batch of data first."
            )
        if self.__cpp_index is None:
            self.__cpp_index = self.__index_py.get_cpp_index()
        if self.__cpp_index is None:
            raise RuntimeError(
                "Collection index backend is unavailable. Call insert() with the first batch of data first."
            )
        return self.__cpp_index

    def _maybe_persist_schema_for_recovery(self) -> None:
        if not self.__index_params.rocksdb_path or self.__index_py is None:
            return

        collection_dir = os.path.dirname(self.__index_params.rocksdb_path)
        if not collection_dir:
            return

        save_schema(
            os.path.join(collection_dir, "schema.json"),
            {"type": "collection", "index": self.__index_params.to_json_dict()},
        )

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
        _assert(num_threads > 0, "num_threads must be greater than 0")
        _assert(ef_search >= limit, "ef_search must be greater than or equal to limit")

        vectors_arr, _ = _validate_query_vectors(vectors, self.__index_py.get_dim(), metric=self.__index_params.metric)

        # Query validation already normalizes cosine queries before they reach the native index.
        ids_arr, dists_arr = self.__index_py.batch_search_with_distance(
            vectors_arr,
            limit,
            ef_search,
            num_threads,
        )

        cpp_index = self._get_cpp_index()
        flat_scalars = cpp_index.batch_get_scalar_data_by_internal_ids(ids_arr.reshape(-1))
        scalar_offset = 0
        ret = {"id": [], "document": [], "metadata": [], "distance": []}
        for ids_row, dists_row in zip(ids_arr, dists_arr):
            row_ids = []
            row_docs = []
            row_metas = []
            row_dists = [float(d) for d in dists_row]

            for _ in ids_row:
                scalar = flat_scalars[scalar_offset]
                scalar_offset += 1
                row_ids.append(scalar.get("item_id", ""))
                row_docs.append(scalar.get("document", ""))
                row_metas.append(scalar.get("metadata", {}))

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
        metadata_filter: Optional[Union[dict, _MetadataFilter]] = None,
        ef_search: int = 100,
        num_threads: int = 1,
        filter_execution_hint: Optional[str] = None,
    ) -> dict:
        """
        Queries the index using vectors with metadata filtering.

        Args:
            vectors: Query vectors.
            limit: Result size per query.
            metadata_filter: Metadata predicate.
            ef_search: ANN ef parameter.
            num_threads: Thread count for batch query.
            filter_execution_hint: Optional hybrid-search execution hint.
                Supported values: ``None``/``"auto"``, ``"disable"``,
                ``"bitset_prefilter"``, ``"iterative_filter"``.

        Returns:
            dict with a single key ``id`` containing item-id rows for each query.
        """
        _assert(self.__index_py is not None, "Index is not initialized yet")
        _assert(ef_search >= limit, "ef_search must be >= limit")
        _assert(num_threads > 0, "num_threads must be greater than 0")

        cpp_index = self._get_cpp_index()
        filter_obj = self._build_filter(metadata_filter)
        filter_hint = normalize_filter_execution_hint(filter_execution_hint)
        vectors_arr, is_single_query = _validate_query_vectors(
            vectors,
            self.__index_py.get_dim(),
            metric=self.__index_params.metric,
        )

        if is_single_query:
            _, item_ids = cpp_index.hybrid_search(
                vectors_arr[0],
                limit,
                ef_search,
                filter_obj,
                bf=False,
                filter_execution_hint=filter_hint,
            )
            return {"id": [list(item_ids)]}
        else:
            _, item_ids = cpp_index.batch_hybrid_search(
                vectors_arr,
                limit,
                ef_search,
                filter_obj,
                num_threads,
                bf=False,
                filter_execution_hint=filter_hint,
            )
            return {"id": [list(row) for row in item_ids]}

    def filter_query(self, metadata_filter: Union[dict, _MetadataFilter], limit: int = 100) -> dict:
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
            dt = valid_dtype(np.asarray(first_embedding).dtype)
            self.__index_params.data_type = dt

            self.__index_params.fill_none_values()

            # Collection always requires scalar data storage
            self.__index_params.has_scalar_data = True

            index = Index(self.__name, self.__index_params)

            # Prepare batch data
            vectors = np.array([item[2] for item in items], dtype=dt)
            item_ids = [item[0] for item in items]
            documents = [item[1] for item in items]
            metadata_list = [item[3] for item in items]

            # Fit with scalar data
            build_threads = self.__index_params.build_threads
            if build_threads is None:
                build_threads = 1
            else:
                _assert(build_threads > 0, "index_params.build_threads must be greater than 0")
            index.fit(
                vectors,
                ef_construction=400,
                num_threads=build_threads,
                item_ids=item_ids,
                documents=documents,
                metadata_list=metadata_list,
            )
            self.__index_py = index
            self.__cpp_index = self.__index_py.get_cpp_index()
            self._maybe_persist_schema_for_recovery()
        else:
            # Incremental insert with scalar data
            cpp_index = self._get_cpp_index()
            for item_id, document, embedding, metadata in items:
                vec = np.array(embedding, dtype=self.__index_py.get_dtype())
                vec = normalize_vectors_for_cosine_metric(vec, self.__index_params.metric)
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
        for item_id, document, embedding, metadata in items:
            vec = np.array(embedding, dtype=self.__index_py.get_dtype())
            vec = normalize_vectors_for_cosine_metric(vec, self.__index_params.metric)
            cpp_index.upsert(
                vec,
                100,  # ef
                item_id,
                document,
                metadata or {},
            )

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

    def reindex(self, ef_construction: int = 400, num_threads: Optional[int] = None):
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
        if num_threads is None:
            rebuild_threads = self.__index_params.build_threads
            if rebuild_threads is None:
                rebuild_threads = 1
            else:
                _assert(rebuild_threads > 0, "index_params.build_threads must be greater than 0")
        else:
            _assert(num_threads > 0, "num_threads must be greater than 0")
            rebuild_threads = num_threads
        self.__index_py.fit(
            vectors,
            ef_construction=ef_construction,
            num_threads=rebuild_threads,
            item_ids=item_ids,
            documents=documents,
            metadata_list=metadata_list,
        )
        self.__cpp_index = self.__index_py.get_cpp_index()

    def _build_filter(self, filter_dict: Optional[Union[dict, _MetadataFilter]]) -> _MetadataFilter:
        """
        Convert Python dict to C++ MetadataFilter.
        """
        mf = _MetadataFilter()
        if filter_dict is None:
            return mf
        if isinstance(filter_dict, _MetadataFilter):
            return filter_dict

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
                    # TODO(review - filter DSL parity): expose `$ne`, `$not_in`, and `$contains`
                    # here so the Python dict DSL stays in sync with every C++ `FilterOp`.
                    if op == "$eq":
                        mf.add_eq(key, op_value)
                    elif op == "$gt":
                        mf.add_gt(key, op_value)
                    elif op == "$ge":
                        mf.add_ge(key, op_value)
                    elif op == "$lt":
                        mf.add_lt(key, op_value)
                    elif op == "$le":
                        mf.add_le(key, op_value)
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
            index_params.rocksdb_path = _default_rocksdb_path(name, url)

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
        cpp_index = self.__cpp_index
        if cpp_index is None and self.__index_py is not None:
            cpp_index = self.__index_py.get_cpp_index()
        if cpp_index is not None:
            cpp_index.close_db()
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
