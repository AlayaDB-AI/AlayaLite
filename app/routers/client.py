import sys

from alayalite import Client
from fastapi import APIRouter

from app.models.collection import (
    CreateCollectionRequest,
    DeleteByFilterRequest,
    DeleteByIdRequest,
    DeleteCollectionRequest,
    InsertCollectionRequest,
    QueryCollectionRequest,
    UpsertCollectionRequest,
)

router = APIRouter()
client = Client()


@router.post(path="/collection/create", tags=["collection"])
async def create_collection(request: CreateCollectionRequest):
    try:
        client.create_collection(request.collection_name)
        return f"Collection {request.collection_name} created successfully"
    except Exception as e:
        print(e, file=sys.stderr)
        raise e


@router.post(path="/collection/list", tags=["collection"])
async def list_collections():
    try:
        collections: list[str] = list(client.list_collections())
        return collections
    except Exception as e:
        print(e, file=sys.stderr)
        raise e


@router.post(path="/collection/delete", tags=["collection"])
async def delete_collection(request: DeleteCollectionRequest):
    try:
        client.delete_collection(request.collection_name)
        return f"Collection {request.collection_name} deleted successfully"
    except Exception as e:
        print(e, file=sys.stderr)
        raise e


@router.post(path="/collection/reset", tags=["collection"])
async def reset_collection():
    try:
        client.reset()
        return "Collection reset successfully"
    except Exception as e:
        print(e, file=sys.stderr)
        raise e


@router.post(path="/collection/insert", tags=["collection"])
async def insert_collection(request: InsertCollectionRequest):
    try:
        collection = client.get_collection(request.collection_name)
        if collection is None:
            raise Exception(f"Collection {request.collection_name} does not exist")
        collection.insert(request.items)
        return f"Successfully inserted {len(request.items)} items into collection {request.collection_name}"
    except Exception as e:
        print(e, file=sys.stderr)
        raise e


@router.post(path="/collection/query", tags=["collection"])
async def query_collection(request: QueryCollectionRequest):
    try:
        collection = client.get_collection(request.collection_name)
        if collection is None:
            raise Exception(f"Collection {request.collection_name} does not exist")
        result = collection.batch_query(
            request.query_vector,
            limit=request.limit,
            ef_search=request.ef_search,
            num_threads=request.num_threads,
        )
        return result
    except Exception as e:
        print(e, file=sys.stderr)
        raise e


@router.post(path="/collection/upsert", tags=["collection"])
async def upsert_collection(request: UpsertCollectionRequest):
    try:
        collection = client.get_collection(request.collection_name)
        if collection is None:
            raise Exception(f"Collection {request.collection_name} does not exist")
        collection.upsert(request.items)
        return f"Successfully upserted {len(request.items)} items into collection {request.collection_name}"
    except Exception as e:
        print(e, file=sys.stderr)
        raise e


@router.post(path="/collection/delete_by_id", tags=["collection"])
async def delete_by_id(request: DeleteByIdRequest):
    try:
        collection = client.get_collection(request.collection_name)
        if collection is None:
            raise Exception(f"Collection {request.collection_name} does not exist")
        collection.delete_by_id(request.ids)
        return f"Successfully deleted {len(request.ids)} items from collection {request.collection_name}"
    except Exception as e:
        print(e, file=sys.stderr)
        raise e


@router.post(path="/collection/delete_by_filter", tags=["collection"])
async def delete_by_filter(request: DeleteByFilterRequest):
    try:
        collection = client.get_collection(request.collection_name)
        if collection is None:
            raise Exception(f"Collection {request.collection_name} does not exist")
        collection.delete_by_filter(request.filter)
        return f"Successfully deleted {len(request.filter)} items from collection {request.collection_name}"
    except Exception as e:
        print(e, file=sys.stderr)
        raise e
