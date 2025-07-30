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
"""

import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd

from .common import _assert
from .index import Index
from .schema import IndexParams, load_schema


# pylint: disable=unused-private-member
class Collection:
    """
    @brief Collection class to manage a collection of documents and their embeddings.
    """

    def __init__(self, name: str):
        """
        Initializes the collection.

        Args:
            name (str): The name of the collection.
        """
        self.__name = name
        self.__dataframe = pd.DataFrame(columns=["id", "document", "metadata"])
        self.__index_py = None
        self.__outer_inner_map = {}
        self.__inner_outer_map = {}

    def batch_query(self, vectors: List[List[float]], limit: int, ef_search: int = 100, num_threads: int = 1) -> dict:
        """
        Queries the index using a batch of vectors.
        """
        _assert(self.__index_py is not None, "Index is not initialized yet")
        _assert(len(vectors) > 0, "vectors must not be empty")
        _assert(
            len(vectors[0]) == self.__index_py.get_dim(),
            "Vector dimension must match the index dimension.",
        )
        _assert(num_threads > 0, "num_threads must be greater than 0")
        _assert(ef_search >= limit, "ef_search must be greater than or equal to limit")

        all_results, all_distance = self.__index_py.batch_search_with_distance(
            np.array(vectors, dtype=np.float32), limit, ef_search, num_threads
        )

        ret = {"id": [], "document": [], "metadata": [], "distance": []}
        for ids, distances in zip(all_results, all_distance):
            uuids = [self.__inner_outer_map.get(idx) for idx in ids if idx in self.__inner_outer_map]
            if not uuids:
                ret["id"].append([])
                ret["document"].append([])
                ret["metadata"].append([])
                ret["distance"].append([])
                continue

            temp_df = self.__dataframe[self.__dataframe["id"].isin(uuids)]
            # Preserve the order of results from the vector search
            temp_df = temp_df.set_index("id").loc[uuids].reset_index()

            df_dict = temp_df.to_dict("list")
            ret["id"].append(df_dict["id"])
            ret["document"].append(df_dict["document"])
            ret["metadata"].append(df_dict["metadata"])
            ret["distance"].append(distances.tolist())
        return ret

    def filter_query(self, metadata_filter: dict, limit: Optional[int] = None) -> dict:
        """
        Filters the DataFrame based on metadata conditions.
        """
        mask = self.__dataframe["metadata"].apply(lambda x: all(x.get(k) == v for k, v in metadata_filter.items()))
        filtered_df = self.__dataframe[mask]

        if limit is not None:
            filtered_df = filtered_df.head(limit)

        return filtered_df.to_dict(orient="list")

    def insert(self, items: List[tuple]):
        """
        Inserts multiple documents and their embeddings into the collection.
        """
        if not items:
            return

        if self.__index_py is None:
            _, _, first_embedding, _ = items[0]
            params = IndexParams(data_type=np.array(first_embedding).dtype, metric="l2")
            self.__index_py = Index(self.__name, params)
            all_embeddings = np.array([item[2] for item in items], dtype=params.data_type)
            self.__index_py.fit(all_embeddings, ef_construction=100, num_threads=1)

            new_rows = []
            for i, (item_id, document, _, metadata) in enumerate(items):
                new_rows.append({"id": item_id, "document": document, "metadata": metadata})
                self.__outer_inner_map[item_id] = i
                self.__inner_outer_map[i] = item_id
            self.__dataframe = pd.concat([self.__dataframe, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            new_rows = []
            for item_id, document, embedding, metadata in items:
                new_rows.append({"id": item_id, "document": document, "metadata": metadata})
                index_id = self.__index_py.insert(np.array(embedding, dtype=self.__index_py.get_dtype()))
                self.__outer_inner_map[item_id] = index_id
                self.__inner_outer_map[index_id] = item_id
            self.__dataframe = pd.concat([self.__dataframe, pd.DataFrame(new_rows)], ignore_index=True)

    def upsert(self, items: List[tuple]):
        """
        Inserts new items or updates existing ones.
        """
        if not items:
            return

        if self.__index_py is None:
            self.insert(items)
            return

        new_items_to_insert = []
        for item_id, document, embedding, metadata in items:
            if item_id in self.__outer_inner_map:
                # Update existing item
                inner_id = self.__outer_inner_map[item_id]
                self.__index_py.remove(inner_id)
                new_index_id = self.__index_py.insert(np.array(embedding, dtype=self.__index_py.get_dtype()))
                self.__outer_inner_map[item_id] = new_index_id
                self.__inner_outer_map[new_index_id] = item_id
                # Update DataFrame
                self.__dataframe.loc[self.__dataframe["id"] == item_id, ["document", "metadata"]] = [document, metadata]
            else:
                # This is a new item, add to list for batch insertion
                new_items_to_insert.append((item_id, document, embedding, metadata))

        if new_items_to_insert:
            self.insert(new_items_to_insert)

    def delete_by_id(self, ids: List[str]):
        """
        Deletes documents from the collection by their IDs.
        """
        if not ids:
            return

        # Remove from DataFrame
        self.__dataframe = self.__dataframe[~self.__dataframe["id"].isin(ids)]

        # Remove from index and maps
        for item_id in ids:
            if item_id in self.__outer_inner_map:
                inner_id = self.__outer_inner_map[item_id]
                self.__index_py.remove(inner_id)
                del self.__outer_inner_map[item_id]
                del self.__inner_outer_map[inner_id]

    def get_by_id(self, ids: List[str]) -> dict:
        """
        Gets documents from the collection by their IDs.
        """
        if not ids:
            return {"id": [], "document": [], "metadata": []}
        return self.__dataframe[self.__dataframe["id"].isin(ids)].to_dict("list")

    def delete_by_filter(self, metadata_filter: dict):
        """
        Deletes items from the collection based on a metadata filter.
        """
        mask = self.__dataframe["metadata"].apply(lambda x: all(x.get(k) == v for k, v in metadata_filter.items()))
        ids_to_delete = self.__dataframe[mask]["id"].tolist()
        if ids_to_delete:
            self.delete_by_id(ids_to_delete)

    def save(self, url):
        """
        Saves the collection to disk.
        """
        if not os.path.exists(url):
            os.makedirs(url)

        data_url = os.path.join(url, "collection.pkl")
        data = {
            "dataframe": self.__dataframe,
            "outer_inner_map": self.__outer_inner_map,
            "inner_outer_map": self.__inner_outer_map,
        }
        with open(data_url, "wb") as f:
            pickle.dump(data, f)

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

        instance = cls(name)
        collection_data_url = os.path.join(collection_url, "collection.pkl")
        with open(collection_data_url, "rb") as f:
            collection_data = pickle.load(f)
            instance.__dataframe = collection_data["dataframe"]
            instance.__outer_inner_map = collection_data["outer_inner_map"]
            instance.__inner_outer_map = collection_data["inner_outer_map"]

        instance.__index_py = Index.load(url, name)
        return instance
