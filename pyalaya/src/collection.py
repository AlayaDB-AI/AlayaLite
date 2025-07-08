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

import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd

from .common import _assert
from .index import Index
from .schema import IndexParams, load_schema


class Collection:
    """
    @brief Collection class to manage a collection of documents and their embeddings.

    """

    def __init__(self, name: str):
        """
        Initializes the collection with an empty DataFrame and mapping structures.

        Args:
            name (str): The name of the collection.
        """
        self.__name = name
        self.__dataframe = pd.DataFrame(columns=["id", "document", "metadata"])
        self.__dataframe.set_index("id", inplace=True)

        self.__index_py = None  # Index object in python for vector accessing
        self.__outer_inner_map = {}  # outer_id (id) -> index_id
        self.__inner_outer_map = {}  # index_id -> id

    def batch_query(
        self, vectors: list[list[float | int]], limit: int, ef_search: int = 100, num_threads: int = 1
    ) -> pd.DataFrame:
        """
        Queries the index using a batch of vectors and retrieves the nearest documents.

        Args:
            vectors (List[List[float]]): A list of query vectors.
            limit (int): The number of nearest neighbors to retrieve per query.
            ef_search (int, optional): Search parameter controlling the trade-off between accuracy and speed. Default is 100.
            num_threads (int, optional): Number of threads for search. Default is 1.

        Returns:
            pd.DataFrame: A DataFrame containing the retrieved documents and their metadata.

        Example:
            >>> collection.batch_query([[0.1, 0.2, 0.3]], limit=2, ef_search=10, num_threads=1)
            {'id': [[1.0, 2.0]], 'document': [['Document 1', 'Document 2']], 'metadata': [[{'category': 'A'}, {'category': 'B'}]], 'distance': [[0.0, 0.27000001072883606]]}

        """
        _assert(self.__index_py is not None, "Index is not init yet")
        _assert(len(vectors) > 0, "vectors must not be empty")
        _assert(
            len(vectors[0]) == self.__index_py.get_dim(),
            "vectors dimension must match the dimension of the vectors used to fit the index.",
        )
        _assert(num_threads > 0, "num_threads must be greater than 0")
        _assert(ef_search > limit, "ef_search must be greater than limit")

        # 2D array: (query_num, k)
        all_results, all_distance = self.__index_py.batch_search_with_distance(
            np.array(vectors), limit, ef_search, num_threads
        )

        ret = {"id": [], "document": [], "metadata": [], "distance": []}
        for ids, distances in zip(all_results, all_distance):
            df = self.__dataframe.iloc[ids].to_dict(orient="list")
            ret["id"].append(df["id"])
            ret["document"].append(df["document"])
            ret["metadata"].append(df["metadata"])
            ret["distance"].append(distances.tolist())
        return ret

    def filter_query(self, filter: dict, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Filter the DataFrame based on given conditions and return a subset of rows.

        Args:
            filter (dict): A dictionary storing the filter conditions on metadata.
            limit (Optional[int]): Maximum number of rows to return. If None, returns all matching rows.
        Returns:
            pd.DataFrame: A filtered copy of the original DataFrame matching all conditions.
        Example:
            >>> collection.filter_query({'category': ['A', 'B'], 'status': 1}, limit=10)
            Returns first 10 rows where category is either 'A' or 'B' AND status equals 1 in metadata column.
        """
        mask = self.__dataframe["metadata"].apply(lambda x: all(x.get(k) == v for k, v in filter.items()))
        filtered_df = self.__dataframe[mask]

        if limit is not None:
            filtered_df = filtered_df.head(limit)

        return filtered_df.to_dict(orient="list")

    # List of (id, document, embedding, metadata)
    def insert(self, items: List[tuple]):
        """
        Inserts multiple documents and their embeddings into the collection.

        Args:
            items (List[tuple]): List of tuples containing (id, document, embedding, metadata).
        """
        if self.__index_py is None:
            _, _, embedding, _ = items[0]

            params = IndexParams(data_type=np.array(embedding).dtype, metric="l2")

            self.__index_py = Index(self.__name, params)
            self.__index_py.fit(np.array([item[2] for item in items]), ef_construction=100, num_threads=1)
            print(type(items[0][0]))
            for i, (id, document, _, metadata) in enumerate(items):
                self.__dataframe = pd.concat(
                    [self.__dataframe, pd.DataFrame([{"id": id, "document": document, "metadata": metadata}])],
                    ignore_index=True,
                )

                self.__outer_inner_map[id] = i  # the inner index is start from 0, and increase by 1 when fitting.
                self.__inner_outer_map[i] = id
        else:
            for item in items:
                (
                    id,
                    document,
                    embedding,
                    metadata,
                ) = item
                self.__dataframe = pd.concat(
                    [self.__dataframe, pd.DataFrame([{"id": id, "document": document, "metadata": metadata}])],
                    ignore_index=True,
                )

                index_id = self.__index_py.insert(np.array(embedding))
                self.__outer_inner_map[id] = index_id
                self.__inner_outer_map[index_id] = id

    # List of (id, document, metadata, distance)
    def upsert(self, items: List[tuple]):
        """
        Inserts new items into the collection or updates existing items if they already exist.

        Args:
            items (List[tuple]): A list of tuples containing (id, document, embedding, metadata).
        """
        if self.__index_py is None:
            self.__index_py = Index(self.__name, IndexParams())
            self.__index_py.fit(np.array([item[2] for item in items]), ef_construction=100, num_threads=1)
            for i in range(len(items)):
                id, document, embedding, metadata = items[i]
                self.__dataframe = pd.concat(
                    [self.__dataframe, pd.DataFrame([{"id": id, "document": document, "metadata": metadata}])],
                    ignore_index=True,
                )

                self.__outer_inner_map[id] = i  # the inner index is start from 0, and increase by 1 when fitting.
                self.__inner_outer_map[i] = id
        else:
            for item in items:
                id, document, embedding, metadata = item
                if id in self.__outer_inner_map:
                    self.__index_py.remove(self.__outer_inner_map[id])
                    new_index_id = self.__index_py.insert(np.array(embedding))
                    self.__outer_inner_map[id] = new_index_id
                    self.__inner_outer_map[new_index_id] = id
                    self.__dataframe.loc[self.__dataframe["id"] == id, ["id", "document", "metadata"]] = [
                        id,
                        document,
                        metadata,
                    ]
                else:
                    self.__dataframe = pd.concat(
                        [self.__dataframe, pd.DataFrame([{"id": id, "document": document, "metadata": metadata}])],
                        ignore_index=True,
                    )

                    index_id = self.__index_py.insert(embedding)
                    self.__outer_inner_map[id] = index_id
                    self.__inner_outer_map[index_id] = id

    def delete_by_id(self, ids: List[str]):
        """
        Deletes documents from the collection by their IDs.

        Args:
            ids (List[str]): List of document IDs to delete.
        """
        self.__dataframe = self.__dataframe[~self.__dataframe["id"].isin(ids)]
        for id in ids:
            if id in self.__outer_inner_map:
                self.__index_py.remove(self.__outer_inner_map[id])
                inner_id = self.__outer_inner_map[id]
                del self.__outer_inner_map[id]
                del self.__inner_outer_map[inner_id]

    def get_by_id(self, ids: List[str]):
        """
        Get documents from the collection by their IDs.

        Args:
            ids (List[str]): List of document IDs to get.

        Example:
            >>> collection.get_by_id([1])
            {'document': ['Document 1'], 'metadata': [{'category': 'A'}], 'id': [1.0]}
        """
        return self.__dataframe[self.__dataframe["id"].isin(ids)].to_dict(orient="list")

    def delete_by_filter(self, filter: dict):
        """
        Deletes items from the collection based on a metadata filter.

        Args:
            filter (dict): A dictionary storing the filter conditions on metadata.
        """
        mask = self.__dataframe["metadata"].apply(lambda x: all(x.get(k) == v for k, v in filter.items()))
        for _, row in self.__dataframe[mask].iterrows():
            inner_id = self.__outer_inner_map[row["id"]]
            self.__index_py.remove(self.__outer_inner_map[row["id"]])
            del self.__outer_inner_map[row["id"]]
            del self.__inner_outer_map[inner_id]
        self.__dataframe = self.__dataframe[~mask]

    def save(self, url):
        """
        Saves the collection to disk.

        Args:
            url (str): Directory path to save the collection.
        """
        if not os.path.exists(url):
            os.makedirs(url)

        data_url = os.path.join(url, "collection.pkl")
        data = {
            "dataframe": self.__dataframe.to_dict(orient="list"),
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

        Args:
            url (str): Directory path where the collection is stored.
            name (str): Collection name.

        Returns:
            Collection: Loaded collection instance.
        """
        collection_url = os.path.join(url, name)

        if not os.path.exists(collection_url):
            raise RuntimeError(f"Collection {name} does not exist")

        schema_url = os.path.join(collection_url, "schema.json")
        schema_map = load_schema(schema_url)

        if not schema_map["type"] or schema_map["type"] != "collection":
            raise RuntimeError(f"{name} is not a collection")

        instance = cls(name)
        collection_data_url = os.path.join(collection_url, "collection.pkl")
        with open(collection_data_url, "rb") as f:
            collection_data = pickle.load(f)
            instance.__dataframe = pd.DataFrame(collection_data["dataframe"])
            instance.__outer_inner_map = collection_data["outer_inner_map"]
            instance.__inner_outer_map = collection_data["inner_outer_map"]

        instance.__index_py = Index.load(url, name)

        return instance
