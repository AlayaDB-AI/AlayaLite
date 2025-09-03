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
This module defines the schema for indices and collections, including
parameter classes and functions for saving and loading schema files.
"""

import json
import os
import shutil
from dataclasses import dataclass

import numpy as np

from ._alayalitepy import IndexParams as _IndexParams
from .common import (
    IDType,
    VectorDType,
    assert_valid_index_type,
    assert_valid_metric_type,
    assert_valid_quantization_type,
    valid_capacity_type,
    valid_dtype,
    valid_id_type,
    valid_index_type,
    valid_max_nbrs,
    valid_metric_type,
    valid_quantization_type,
)

__all__ = ["IndexParams", "load_schema", "save_schema"]


@dataclass
class IndexParams:
    """Parameters for configuring vector index creation and management."""

    index_type: str = None
    data_type: VectorDType = None
    id_type: IDType = None
    quantization_type: str = None
    metric: str = None
    capacity: np.uint32 = None
    max_nbrs: int = None

    def index_path(self, folder_uri):
        return os.path.join(folder_uri, f"{self.index_type}_{self.metric}_{self.max_nbrs}.index")

    def data_path(self, folder_uri):
        return os.path.join(folder_uri, "raw.data")

    def quant_path(self, folder_uri):
        if self.quantization_type == "none":
            return ""
        else:
            return os.path.join(folder_uri, f"{self.quantization_type}.data")

    def fill_none_values(self):
        if self.index_type is None:
            self.index_type = "hnsw"
        if self.data_type is None:
            self.data_type = np.float32
        if self.id_type is None:
            self.id_type = np.uint32
        if self.quantization_type is None:
            self.quantization_type = "none"
        if self.metric is None:
            self.metric = "l2"
        if self.capacity is None:
            self.capacity = 100000
        if self.max_nbrs is None:
            self.max_nbrs = 32

    def to_cpp_params(self):
        native_index_type = valid_index_type(self.index_type)
        native_data_type = valid_dtype(self.data_type)
        native_id_type = valid_id_type(self.id_type)
        native_metric_type = valid_metric_type(self.metric)
        native_quantization_type = valid_quantization_type(self.quantization_type)
        capacity = valid_capacity_type(self.capacity)

        return _IndexParams(
            index_type_=native_index_type,
            data_type_=native_data_type,
            id_type_=native_id_type,
            quantization_type_=native_quantization_type,
            metric_=native_metric_type,
            capacity_=capacity,
        )

    def to_json_dict(self) -> str:
        return {
            "index_type": self.index_type,
            "data_type": np.dtype(self.data_type).name,  # Convert dtype to string
            "id_type": np.dtype(self.id_type).name,  # Convert dtype to string
            "quantization_type": self.quantization_type,
            "metric": self.metric,
            "capacity": self.capacity,
            "max_nbrs": self.max_nbrs,
        }

    @classmethod
    def from_str_dict(cls, data: str) -> "IndexParams":
        """Deserialize from a JSON string."""
        return cls(
            index_type=data["index_type"],
            data_type=np.dtype(data["data_type"]).type,  # Convert back to dtype
            id_type=np.dtype(data["id_type"]).type,  # Convert back to dtype
            quantization_type=data["quantization_type"],
            metric=data["metric"],
            capacity=data["capacity"],
            max_nbrs=data["max_nbrs"],
        )

    @classmethod
    def from_kwargs(cls, **kwargs) -> "IndexParams":
        index_type = None
        data_type = None
        id_type = None
        quantization_type = None
        metric = None
        capacity = None
        max_nbrs = None

        if kwargs.get("index_type") is not None:
            ind_type = kwargs.get("index_type")
            assert_valid_index_type(ind_type)
            index_type = ind_type
        if kwargs.get("data_type") is not None:
            data_type = valid_dtype(kwargs.get("data_type"))
        if kwargs.get("id_type") is not None:
            id_type = valid_id_type(kwargs.get("id_type"))
        if kwargs.get("quantization_type") is not None:
            qt = kwargs.get("quantization_type")
            assert_valid_quantization_type(qt)
            quantization_type = qt
        if kwargs.get("metric") is not None:
            mt = kwargs.get("metric")
            assert_valid_metric_type(mt)
            metric = mt
        if kwargs.get("capacity") is not None:
            capacity = valid_capacity_type(kwargs.get("capacity"))
        if kwargs.get("max_nbrs") is not None:
            max_nbrs = valid_max_nbrs(kwargs.get("max_nbrs"))
        return cls(
            index_type=index_type,
            data_type=data_type,
            id_type=id_type,
            quantization_type=quantization_type,
            metric=metric,
            capacity=capacity,
            max_nbrs=max_nbrs,
        )


@classmethod
class RAGParams:
    """Parameters for configuring Retrieval-Augmented Generation (RAG) processes."""

    chunker: str = "fix_size"  # [fix_size, semantic, sentence]
    chunk_size: int = 1024
    chunk_overlap: int = 0
    semantic_model: str = "all-MiniLM-L6-v2"

    embedder: str = "bge"  # [bge, m3e, multilingual, jina]
    embedder_model_path: str = ""


def load_schema(url) -> dict:
    if not os.path.exists(url):
        raise FileNotFoundError("The schema file does not exist!")
    with open(url, encoding="utf-8") as f:
        return json.load(f)


def save_schema(schema_url, schema_map):
    schema_bak_address = schema_url + ".bak"
    shutil.copy2(schema_url, schema_bak_address)
    with open(schema_url, "w", encoding="utf-8") as f:
        json.dump(schema_map, f, indent=4)
        os.remove(schema_bak_address)


def is_index_url(url):
    schema_url = os.path.join(url, "schema.json")
    if not os.path.exists(schema_url):
        return False
    else:
        schema_map = load_schema(schema_url)
        return schema_map["type"] == "index"


def is_collection_url(url):
    schema_url = os.path.join(url, "schema.json")
    if not os.path.exists(schema_url):
        return False
    else:
        schema_map = load_schema(schema_url)
        return schema_map["type"] == "collection"
