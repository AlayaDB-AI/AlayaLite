from pydantic import BaseModel

class CreateCollectionRequest(BaseModel):
    collection_name: str

class DeleteCollectionRequest(BaseModel):
    collection_name: str