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
This module defines common types, constants, and validation functions used throughout the alayalite library.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Type, Union

import numpy as np
from numpy import typing as npt

from ._alayalitepy import IndexType as _IndexType
from ._alayalitepy import MetricType as _MetricType
from ._alayalitepy import QuantizationType as _QuantizationType

# TypeAlias is only available in Python 3.10+
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    TypeAlias = type  # Fallback for older Python versions

IDType: TypeAlias = Union[Type[np.uint64], Type[np.uint32]]
""" Type alias for one of {`numpy.uint64`, `numpy.uint32`} """
# pylint: disable=invalid-name
VectorDType: TypeAlias = Union[
    Type[np.float32],
    Type[np.int8],
    Type[np.uint8],
    Type[np.float64],
    Type[np.int32],
    Type[np.uint32],
]
""" Type alias for one of {`numpy.float32`, `numpy.int8`, `numpy.uint8`} """
DistanceMetric: TypeAlias = Literal["euclidean", "l2", "ip", "cosine", "cos"]
""" Type alias for one of {"euclidean", "l2", "ip", "cosine", "cos"} """
QuantizationType: TypeAlias = Literal[None, "none", "sq8", "sq4", "rabitq"]
""" Type alias for one of {None, "none", "sq8", "sq4"} """
IndexType: TypeAlias = Literal["hnsw", "nsg", "fusion"]
""" Type alias for one of {"hnsw", "nsg" ,"fusion"} """
VectorLike: TypeAlias = npt.NDArray[VectorDType]  # type: ignore
""" Type alias for something that can be treated as a vector """
VectorLikeBatch: TypeAlias = npt.NDArray[VectorDType]  # type: ignore
""" Type alias for a batch of VectorLikes """

_VALID_IDTYPES = [np.uint64, np.uint32]
_VALID_DTYPES = [np.float32, np.int8, np.uint8, np.float64, np.int32, np.uint32]
_VALID_METRIC_TYPES = ["euclidean", "l2", "ip", "cosine", "cos"]
_VALID_INDEX_TYPES = ["hnsw", "nsg", "fusion"]
_VALID_SQ_TYPES = [None, "none", "sq8", "sq4", "rabitq"]

__all__ = [
    "VectorDType",
    "IDType",
    "valid_id_type",
    "valid_dtype",
    "valid_metric_type",
    "valid_index_type",
    "valid_capacity_type",
    "valid_quantization_type",
    "valid_max_nbrs",
]


def valid_dtype(dtype) -> np.dtype:
    _assert(
        any(np.can_cast(dtype, dtype_) for dtype_ in _VALID_DTYPES),
        "Vector dtype must be one of type {(np.single, np.float32), (np.byte, np.int8), "
        "(np.ubyte, np.uint8), (np.double, np.float64), (np.int32, np.int32), (np.uint32, np.uint32)}",
    )
    return np.dtype(dtype)


def valid_id_type(id_type) -> np.dtype:
    _assert(
        any(np.can_cast(id_type, dtype_) for dtype_ in _VALID_IDTYPES),
        "ID dtype must be of one of type {(np.uint64), (np.uint32)}",
    )
    return np.dtype(id_type)


def valid_capacity_type(capacity: np.dtype) -> np.uint32:
    _assert(
        capacity > 0,
        "Capacity must be greater than 0",
    )
    return capacity


def assert_valid_metric_type(metric: str) -> None:
    _assert(
        metric.lower() in _VALID_METRIC_TYPES,
        f"Distance metric must be one of {_VALID_METRIC_TYPES}",
    )


def valid_metric_type(metric: str) -> _MetricType:
    assert_valid_metric_type(metric)
    if metric.lower() == "ip":
        return _MetricType.IP
    elif metric.lower() == "l2" or metric.lower() == "euclidean":
        return _MetricType.L2
    elif metric.lower() == "cosine" or metric.lower() == "cos":
        return _MetricType.COS


def assert_valid_quantization_type(quantization_type: str) -> None:
    _assert(
        quantization_type is None or quantization_type.lower() in _VALID_SQ_TYPES,
        f"Quantization type must be one of {_VALID_SQ_TYPES}",
    )


def valid_quantization_type(quantization_type: str) -> _QuantizationType:
    assert_valid_quantization_type(quantization_type)

    if quantization_type is None:
        return _QuantizationType.NONE
    elif quantization_type.lower() == "none":
        return _QuantizationType.NONE
    elif quantization_type.lower() == "sq8":
        return _QuantizationType.SQ8
    elif quantization_type.lower() == "sq4":
        return _QuantizationType.SQ4
    elif quantization_type.lower() == "rabitq":
        return _QuantizationType.RABITQ


def assert_valid_index_type(index: str) -> None:
    _assert(
        index.lower() in _VALID_INDEX_TYPES,
        f"Index type must be one of {_VALID_INDEX_TYPES}",
    )


def valid_index_type(index: str) -> _IndexType:
    assert_valid_index_type(index)

    if index.lower() == "hnsw":
        return _IndexType.HNSW
    elif index.lower() == "nsg":
        return _IndexType.NSG
    elif index.lower() == "fusion":
        return _IndexType.FUSION


def valid_max_nbrs(max_nbrs: np.uint32) -> np.uint32:
    _assert(
        0 < max_nbrs < 1000,
        "Max neighbors must be greater than 0 and less than 1000",
    )
    return max_nbrs


def valid_index_path(index_path: str) -> None:
    path = Path(index_path)
    _assert(
        path.exists() and path.is_dir(),
        "Index path must be a valid directory",
    )


def valid_index_prefix(index_prefix: str) -> None:
    _assert(
        index_prefix != "",
        "Index prefix must not be empty",
    )


def _assert(statement_eval: bool, message: str) -> None:
    if not statement_eval:
        raise ValueError(message)
