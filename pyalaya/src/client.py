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


import json
import os

from ._alayalitepy import PyIndexInterface as _PyIndexInterface
from .collection import Collection
from .common import (
    valid_capacity_type,
    valid_dtype,
    valid_id_type,
    valid_index_type,
    valid_max_nbrs,
    valid_metric_type,
    valid_quantization_type,
)
from .index import Index
from .schema import IndexParams, is_collection_url, is_index_url

__all__ = ["Client"]


class Client:
    """
    Client manages collections and indices. This class provides methods for
    creating, retrieving, saving, and deleting collections and indices from disk.
    """

    def __init__(self, url=None):
        """
        Initialize the Client. Optionally, provide a URL to load data from disk.
        If no URL is provided, the client cannot save or load any data.

        Args:
            url (str, optional): The directory path from which to load collections and indices. Defaults to None.
        """
        self.__collection_map = {}
        self.__index_map = {}

        if url is not None:
            url = os.path.abspath(url)
            self.__url = url
            if not os.path.exists(url):
                os.makedirs(url)

            print(f"Load AlayaLite data from {url}")
            all_names = [file for file in os.listdir(url) if os.path.isdir(os.path.join(url, file))]
            print(f"{all_names=}")
            for name in all_names:
                full_url = os.path.join(url, name)
                if is_collection_url(full_url):
                    self.__collection_map[name] = Collection.load(url, name)
                    print(f"Collection {name} is loaded")
                elif is_index_url(full_url):
                    self.__index_map[name] = Index.load(url, name)
                    print(f"Index {name} is loaded")
                else:
                    print(f"Unknown url: {full_url} is found")

    def list_collections(self):
        """
        List all collection names currently managed by the client.

        Returns:
            dict_keys: A list of collection names.
        """
        return self.__collection_map.keys()

    def list_indices(self):
        """
        List all index names currently managed by the client.

        Returns:
            dict_keys: A list of index names.
        """
        return self.__index_map.keys()

    def get_collection(self, name: str = "default") -> Collection:
        """
        Get a collection by name. If the collection does not exist, returns None.

        Args:
            name (str, optional): The name of the collection to retrieve. Defaults to "default".

        Returns:
            Collection or None: The collection if found, else None.
        """
        if name in self.__collection_map:
            return self.__collection_map[name]
        else:
            print(f"Collection {name} does not exist")
            return None

    def get_index(self, name: str = "default") -> _PyIndexInterface:
        """
        Get an index by name.

        Args:
            name (str, optional): The name of the index to retrieve. Defaults to "default".

        Returns:
            _PyIndexInterface (cpp class): The index if found.
        """
        return self.__index_map[name]

    def create_collection(self, name: str = "default", **kwargs) -> Collection:
        """
        Create a new collection with the given name and parameters. Raises an error if the collection already exists.

        Args:
            name (str): The name of the collection to create.
            **kwargs: Additional parameters for collection creation.

        Returns:
            Collection: The created collection.

        Raises:
            RuntimeError: If a collection or index with the same name already exists.
        """
        if name in self.__collection_map:
            raise RuntimeError(f"Collection {name} already exists")
        if name in self.__index_map:
            raise RuntimeError(f"Index {name} already exists")

        collection = Collection(name)
        self.__collection_map[name] = collection
        return collection

    def create_index(self, name: str = "default", **kwargs) -> Index:
        """
        Create a new index with the given name and parameters.
        Raises an error if the index already exists.

        Args:
            name (str): The name of the index to create.
            **kwargs: Additional parameters for index creation.

        Returns:
            Index: The created index.

        Raises:
            RuntimeError: If a collection or index with the same name already exists.
        """
        if name in self.__collection_map:
            raise RuntimeError(f"Collection {name} already exists")
        if name in self.__index_map:
            raise RuntimeError(f"Index {name} already exists")

        index_type = None
        data_type = None
        id_type = None
        quantization_type = None
        metric = None
        capacity = None
        max_nbrs = None

        if kwargs.get("index_type") is not None:
            index_type = valid_index_type(kwargs.get("index_type"))
        if kwargs.get("data_type") is not None:
            data_type = valid_dtype(kwargs.get("data_type"))
        if kwargs.get("id_type") is not None:
            id_type = valid_id_type(kwargs.get("id_type"))
        if kwargs.get("quantization_type") is not None:
            quantization_type = valid_quantization_type(kwargs.get("quantization_type"))
        if kwargs.get("metric") is not None:
            metric = valid_metric_type(kwargs.get("metric"))
        if kwargs.get("capacity") is not None:
            capacity = valid_capacity_type(kwargs.get("capacity"))
        if kwargs.get("max_nbrs") is not None:
            max_nbrs = valid_max_nbrs(kwargs.get("max_nbrs"))

        constraints = IndexParams(
            index_type=index_type,
            data_type=data_type,
            id_type=id_type,
            quantization_type=quantization_type,
            metric=metric,
            capacity=capacity,
            max_nbrs=max_nbrs,
        )
        index = Index(name, constraints)
        self.__index_map[name] = index
        return index

    def get_or_create_collection(self, name: str, **kwargs) -> Collection:
        """
        Retrieve a collection if it exists, otherwise create and return a new collection.

        Args:
            name (str): The name of the collection to retrieve or create.
            **kwargs: Parameters for collection creation if it doesn't exist.

        Returns:
            Collection: The existing or newly created collection.
        """
        if name not in self.__collection_map:
            collection = self.create_collection(name, **kwargs)
            self.__collection_map[name] = collection
            return collection
        else:
            return self.__collection_map[name]

    def get_or_create_index(self, name: str, **kwargs) -> Collection:
        """
        Retrieve an index if it exists, otherwise create and return a new index.

        Args:
            name (str): The name of the index to retrieve or create.
            **kwargs: Parameters for index creation if it doesn't exist.

        Returns:
            Index: The existing or newly created index.
        """
        if name not in self.__index_map:
            index = self.create_index(name, **kwargs)
            self.__index_map[name] = index
            return index
        else:
            return self.__index_map[name]

    def delete_collection(self, collection_name: str, delete_on_disk: bool = False):
        """
        Delete a collection by name. Optionally delete it from disk as well.

        Args:
            collection_name (str): The name of the collection to delete.
            delete_on_disk (bool, optional): Whether to delete the collection from disk. Defaults to False.

        Raises:
            RuntimeError: If the collection does not exist.
        """
        if collection_name not in self.__collection_map:
            raise RuntimeError(f"Collection {collection_name} does not exist")
        del self.__collection_map[collection_name]
        if delete_on_disk:
            if self.__url is None:
                raise RuntimeError("Client is not initialized with a url")
            collection_url = os.path.join(self.__url, collection_name)
            if not os.path.exists(collection_url):
                raise RuntimeError(f"Collection {collection_name} does not exist")
            os.rmdir(collection_url)
            print(f"Collection {collection_name} is deleted")

    def delete_index(self, index_name: str, delete_on_disk: bool = False):
        """
        Delete an index by name. Optionally delete it from disk as well.

        Args:
            index_name (str): The name of the index to delete.
            delete_on_disk (bool, optional): Whether to delete the index from disk. Defaults to False.

        Raises:
            RuntimeError: If the index does not exist.
        """
        if index_name not in self.__index_map:
            raise RuntimeError(f"Index {index_name} does not exist")
        del self.__index_map[index_name]
        if delete_on_disk:
            if self.__url is None:
                raise RuntimeError("Client is not initialized with a url")
            index_url = os.path.join(self.__url, index_name)
            if not os.path.exists(index_url):
                raise RuntimeError(f"Index {index_name} does not exist")
            os.rmdir(index_url)
            print(f"Index {index_name} is deleted")

    def reset(self):
        """
        Reset the client by clearing all collections and indices.

        This will remove all loaded data from memory.
        """
        self.__collection_map = {}
        self.__index_map = {}

    def save_index(self, index_name: str):
        """
        Save an index to disk. The index schema will be stored in a JSON file.

        Args:
            index_name (str): The name of the index to save.

        Raises:
            RuntimeError: If the client is not initialized with a URL or the index does not exist.
        """
        if self.__url is None:
            raise RuntimeError("Client is not initialized with a url")
        if index_name not in self.__index_map:
            raise RuntimeError(f"Index {index_name} does not exist")

        index_url = os.path.join(self.__url, index_name)
        if not os.path.exists(index_url):
            os.makedirs(index_url)
        schema_map = self.__index_map[index_name].save(index_name)
        index_schema_url = os.path.join(index_url, "schema.json")
        with open(index_schema_url, "w") as f:
            json.dump(schema_map, f)
        print(f"Index {index_name} is saved")

    def save_collection(self, collection_name: str):
        """
        Save a collection to disk. The collection schema will be stored in a JSON file.

        Args:
            collection_name (str): The name of the collection to save.

        Raises:
            RuntimeError: If the client is not initialized with a URL or the collection does not exist.
        """
        if self.__url is None:
            raise RuntimeError("Client is not initialized with a url")
        if collection_name not in self.__collection_map:
            raise RuntimeError(f"Collection {collection_name} does not exist")
        collection_url = os.path.join(self.__url, collection_name)
        if not os.path.exists(collection_url):
            os.makedirs(collection_url)

        schema_map = self.__collection_map[collection_name].save(collection_url)
        collection_schema_url = os.path.join(collection_url, "schema.json")

        with open(collection_schema_url, "w") as f:
            json.dump(schema_map, f)
        print(f"Collection {collection_name} is saved")
