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
    items: list[tuple[int, str, list[float], dict]]


class UpsertCollectionRequest(BaseModel):
    collection_name: str
    items: list[tuple[int, str, list[float], dict]]


class QueryCollectionRequest(BaseModel):
    collection_name: str
    query_vector: list[list[float]]
    limit: int = 1
    ef_search: int = 10
    num_threads: int = 1


class DeleteByIdRequest(BaseModel):
    collection_name: str
    ids: list[int]


class DeleteByFilterRequest(BaseModel):
    collection_name: str
    filter: dict


class SaveCollectionRequest(BaseModel):
    collection_name: str
