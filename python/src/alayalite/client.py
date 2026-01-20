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
This module provides the main Client class for interacting with the AlayaLite database,
managing indices and collections.
"""

import json
import os
import shutil

from .collection import Collection
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
            url (str, optional): The directory path from which to load data. Defaults to None.
        """
        self.__collection_map = {}
        self.__index_map = {}
        self.__url = None
        if url is not None:
            self.__url = os.path.abspath(url)
            if not os.path.exists(self.__url):
                os.makedirs(self.__url)

            print(f"Load AlayaLite data from {self.__url}")
            all_names = [f for f in os.listdir(self.__url) if os.path.isdir(os.path.join(self.__url, f))]
            print(f"{all_names=}")
            for name in all_names:
                full_url = os.path.join(self.__url, name)
                if is_collection_url(full_url):
                    self.__collection_map[name] = Collection.load(self.__url, name)
                    print(f"Collection {name} is loaded")
                elif is_index_url(full_url):
                    self.__index_map[name] = Index.load(self.__url, name)
                    print(f"Index {name} is loaded")
                else:
                    print(f"Unknown url: {full_url} is found")

    def list_collections(self):
        """
        List all collection names currently managed by the client.

        Returns:
            list: A list of collection names.
        """
        return list(self.__collection_map.keys())

    def list_indices(self):
        """
        List all index names currently managed by the client.

        Returns:
            list: A list of index names.
        """
        return list(self.__index_map.keys())

    def get_collection(self, name: str = "default") -> Collection:
        """
        Get a collection by name. If the collection does not exist, returns None.

        Args:
            name (str, optional): The name of the collection to retrieve. Defaults to "default".

        Returns:
            Collection or None: The collection if found, else None.
        """
        return self.__collection_map.get(name)

    def get_index(self, name: str = "default") -> Index:
        """
        Get an index by name.

        Args:
            name (str, optional): The name of the index to retrieve. Defaults to "default".

        Returns:
            _PyIndexInterface (cpp class): The index if found, else None
        """
        if name in self.__index_map:
            return self.__index_map[name]
        else:
            print(f"Index {name} does not exist")
            return None

    def create_collection(self, name: str = "default", **kwargs) -> Collection:
        """
        Create a new collection with the given name.

        Args:
            name (str): The name of the collection to create.
            **_kwargs: Additional parameters (currently unused).

        Returns:
            Collection: The created collection.

        Raises:
            RuntimeError: If a collection or index with the same name already exists.
        """
        if name in self.__collection_map or name in self.__index_map:
            raise RuntimeError(f"A collection or index with name '{name}' already exists")

        index_params = IndexParams.from_kwargs(**kwargs)
        collection = Collection(name, index_params)
        self.__collection_map[name] = collection
        return collection

    def create_index(self, name: str = "default", **kwargs) -> Index:
        """
        Create a new index with the given name and parameters.

        Args:
            name (str): The name of the index to create.
            **kwargs: Additional parameters for index creation.

        Returns:
            Index: The created index.

        Raises:
            RuntimeError: If a collection or index with the same name already exists.
        """
        if name in self.__collection_map or name in self.__index_map:
            raise RuntimeError(f"A collection or index with name '{name}' already exists")

        params = IndexParams.from_kwargs(**kwargs)
        index = Index(name, params)
        self.__index_map[name] = index
        return index

    def get_or_create_collection(self, name: str, **kwargs) -> Collection:
        """
        Retrieve a collection if it exists, otherwise create a new one.

        Args:
            name (str): The name of the collection to retrieve or create.
            **kwargs: Parameters for collection creation if it doesn't exist.

        Returns:
            Collection: The existing or newly created collection.
        """
        collection = self.get_collection(name)
        if collection is None:
            collection = self.create_collection(name, **kwargs)
        return collection

    def get_or_create_index(self, name: str, **kwargs) -> Index:
        """
        Retrieve an index if it exists, otherwise create a new one.

        Args:
            name (str): The name of the index to retrieve or create.
            **kwargs: Parameters for index creation if it doesn't exist.

        Returns:
            Index: The existing or newly created index.
        """
        index = self.get_index(name)
        if index is None:
            index = self.create_index(name, **kwargs)
        return index

    def delete_collection(self, collection_name: str, delete_on_disk: bool = False):
        """
        Delete a collection by name.

        Args:
            collection_name (str): The name of the collection to delete.
            delete_on_disk (bool, optional): Whether to delete it from disk. Defaults to False.

        Raises:
            RuntimeError: If the collection does not exist or client URL is not set for disk ops.
        """
        if collection_name not in self.__collection_map:
            raise RuntimeError(f"Collection '{collection_name}' does not exist")
        del self.__collection_map[collection_name]
        if delete_on_disk:
            if self.__url is None:
                raise RuntimeError("Client is not initialized with a url for disk operations")
            collection_url = os.path.join(self.__url, collection_name)
            if os.path.exists(collection_url):
                shutil.rmtree(collection_url)
                print(f"Collection '{collection_name}' is deleted from disk")

    def delete_index(self, index_name: str, delete_on_disk: bool = False):
        """
        Delete an index by name.

        Args:
            index_name (str): The name of the index to delete.
            delete_on_disk (bool, optional): Whether to delete it from disk. Defaults to False.

        Raises:
            RuntimeError: If the index does not exist or client URL is not set for disk ops.
        """
        if index_name not in self.__index_map:
            raise RuntimeError(f"Index '{index_name}' does not exist")
        del self.__index_map[index_name]
        if delete_on_disk:
            if self.__url is None:
                raise RuntimeError("Client is not initialized with a url for disk operations")
            index_url = os.path.join(self.__url, index_name)
            if os.path.exists(index_url):
                shutil.rmtree(index_url)
                # TODO: change all print to log
                print(f"Index '{index_name}' is deleted from disk")

    def reset(self, delete_on_disk: bool = False):
        """
        Reset the client
        """
        if delete_on_disk:
            if self.__url is None:
                raise RuntimeError("Client is not initialized with a url for disk operations")

            for collection_name in self.__collection_map:
                index_url = os.path.join(self.__url, collection_name)
                if os.path.exists(index_url):
                    shutil.rmtree(index_url)
                    # logger.info(f'rm {index_url}')

        self.__collection_map = {}
        self.__index_map = {}

    def save_index(self, index_name: str):
        """
        Save an index to disk.

        Args:
            index_name (str): The name of the index to save.

        Raises:
            RuntimeError: If client URL is not set or the index does not exist.
        """
        if self.__url is None:
            raise RuntimeError("Client is not initialized with a url")
        if index_name not in self.__index_map:
            raise RuntimeError(f"Index '{index_name}' does not exist")

        index_url = os.path.join(self.__url, index_name)
        schema_map = self.__index_map[index_name].save(index_url)
        index_schema_url = os.path.join(index_url, "schema.json")
        with open(index_schema_url, "w", encoding="utf-8") as f:
            json.dump(schema_map, f, indent=4)
        print(f"Index '{index_name}' is saved")

    def save_collection(self, collection_name: str):
        """
        Save a collection to disk.

        Args:
            collection_name (str): The name of the collection to save.

        Raises:
            RuntimeError: If client URL is not set or the collection does not exist.
        """
        if self.__url is None:
            raise RuntimeError("Client is not initialized with a url")
        if collection_name not in self.__collection_map:
            raise RuntimeError(f"Collection '{collection_name}' does not exist")

        collection_url = os.path.join(self.__url, collection_name)
        schema_map = self.__collection_map[collection_name].save(collection_url)
        collection_schema_url = os.path.join(collection_url, "schema.json")

        with open(collection_schema_url, "w", encoding="utf-8") as f:
            json.dump(schema_map, f, indent=4)
        print(f"Collection '{collection_name}' is saved")
