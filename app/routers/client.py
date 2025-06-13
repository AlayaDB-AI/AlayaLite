from fastapi import APIRouter
import sys

from alayalite import Client, Collection, Index

from app.models.collection import (
    CreateCollectionRequest,
    DeleteCollectionRequest,
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