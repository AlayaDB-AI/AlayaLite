import os
import sys

from alayalite import Client
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.models.collection import (
    CreateCollectionRequest,
    DeleteByFilterRequest,
    DeleteByIdRequest,
    DeleteCollectionRequest,
    InsertCollectionRequest,
    QueryCollectionRequest,
    SaveCollectionRequest,
    UpsertCollectionRequest,
)

router = APIRouter()

# Storage directory can be configured via the ALAYALITE_DATA_DIR environment variable.
# When running in Docker you can bind-mount a host directory to this path (default: pwd).
storage_dir = os.environ.get("ALAYALITE_DATA_DIR", os.path.abspath("./data"))
if storage_dir:
    # Ensure the directory exists for the client to read/write
    try:
        os.makedirs(storage_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: failed to create storage dir {storage_dir}: {e}", file=sys.stderr)

client = Client(url=storage_dir)


@router.post(path="/collection/create", tags=["collection"])
async def create_collection(request: CreateCollectionRequest):
    try:
        client.create_collection(request.collection_name)
        return f"Collection {request.collection_name} created successfully"
    except Exception as e:
        print(e, file=sys.stderr)
        msg = str(e)
        code = status.HTTP_409_CONFLICT if "already exists" in msg else status.HTTP_400_BAD_REQUEST
        return JSONResponse(status_code=code, content={"error": msg})


@router.post(path="/collection/list", tags=["collection"])
async def list_collections():
    collections: list[str] = list(client.list_collections())
    return collections


@router.post(path="/collection/delete", tags=["collection"])
async def delete_collection(request: DeleteCollectionRequest):
    try:
        client.delete_collection(request.collection_name)
        return f"Collection {request.collection_name} deleted successfully"
    except Exception as e:
        print(e, file=sys.stderr)
        code = status.HTTP_404_NOT_FOUND if "does not exist" in str(e) else status.HTTP_400_BAD_REQUEST
        return JSONResponse(status_code=code, content={"error": str(e)})


@router.post(path="/collection/reset", tags=["collection"])
async def reset_collection():
    client.reset()
    return "Collection reset successfully"


@router.post(path="/collection/insert", tags=["collection"])
async def insert_collection(request: InsertCollectionRequest):
    try:
        collection = client.get_collection(request.collection_name)
        if collection is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": f"Collection {request.collection_name} does not exist"},
            )
        collection.insert(request.items)
        return f"Successfully inserted {len(request.items)} items into collection {request.collection_name}"
    except Exception as e:
        print(e, file=sys.stderr)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(e)})


@router.post(path="/collection/query", tags=["collection"])
async def query_collection(request: QueryCollectionRequest):
    try:
        collection = client.get_collection(request.collection_name)
        if collection is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": f"Collection {request.collection_name} does not exist"},
            )
        result = collection.batch_query(
            request.query_vector,
            limit=request.limit,
            ef_search=request.ef_search,
            num_threads=request.num_threads,
        )
        return result
    except Exception as e:
        print(e, file=sys.stderr)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(e)})


@router.post(path="/collection/upsert", tags=["collection"])
async def upsert_collection(request: UpsertCollectionRequest):
    try:
        collection = client.get_collection(request.collection_name)
        if collection is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": f"Collection {request.collection_name} does not exist"},
            )
        collection.upsert(request.items)
        return f"Successfully upserted {len(request.items)} items into collection {request.collection_name}"
    except Exception as e:
        print(e, file=sys.stderr)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(e)})


@router.post(path="/collection/delete_by_id", tags=["collection"])
async def delete_by_id(request: DeleteByIdRequest):
    try:
        collection = client.get_collection(request.collection_name)
        if collection is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": f"Collection {request.collection_name} does not exist"},
            )
        collection.delete_by_id(request.ids)
        return f"Successfully deleted items from collection {request.collection_name}"
    except Exception as e:
        print(e, file=sys.stderr)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(e)})


@router.post(path="/collection/delete_by_filter", tags=["collection"])
async def delete_by_filter(request: DeleteByFilterRequest):
    try:
        collection = client.get_collection(request.collection_name)
        if collection is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": f"Collection {request.collection_name} does not exist"},
            )
        collection.delete_by_filter(request.filter)
        return f"Successfully deleted {len(request.filter)} items from collection {request.collection_name}"
    except Exception as e:
        print(e, file=sys.stderr)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(e)})


@router.post(path="/collection/save", tags=["collection"])
async def save_collection(request: SaveCollectionRequest):
    try:
        client.save_collection(request.collection_name)
        return f"Collection {request.collection_name} saved successfully"
    except Exception as e:
        print(e, file=sys.stderr)
        code = status.HTTP_404_NOT_FOUND if "does not exist" in str(e) else status.HTTP_400_BAD_REQUEST
        return JSONResponse(status_code=code, content={"error": str(e)})
