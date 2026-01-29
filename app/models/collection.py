from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

__all__ = [
    "SetMetricRequest",
    "CreateCollectionRequest",
    "DeleteCollectionRequest",
    "ResetCollectionRequest",
    "InsertCollectionRequest",
    "UpsertCollectionRequest",
    "QueryCollectionRequest",
    "DeleteByIdRequest",
    "DeleteByFilterRequest",
    "SaveCollectionRequest",
]


class CreateCollectionRequest(BaseModel):
    collection_name: str


class SetMetricRequest(BaseModel):
    collection_name: str
    metric: str


class DeleteCollectionRequest(BaseModel):
    collection_name: str
    delete_on_disk: bool = False


class ResetCollectionRequest(BaseModel):
    delete_on_disk: bool = False


class InsertCollectionRequest(BaseModel):
    collection_name: str
    items: List[Tuple[int, str, List[float], Dict[str, Any]]]


class UpsertCollectionRequest(BaseModel):
    collection_name: str
    items: List[Tuple[int, str, List[float], Dict[str, Any]]]


class QueryCollectionRequest(BaseModel):
    collection_name: str
    query_vector: List[List[float]]
    limit: int = 1
    ef_search: int = 10
    num_threads: int = 1


class DeleteByIdRequest(BaseModel):
    collection_name: str
    ids: List[int]


class DeleteByFilterRequest(BaseModel):
    collection_name: str
    filter: Dict[str, Any]


class SaveCollectionRequest(BaseModel):
    collection_name: str
