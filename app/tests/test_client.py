import pytest
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.mark.asyncio
async def test_create_lists_delete_collection():
    response = client.post("/api/v1/collection/create", json={ "collection_name": "test" })
    print(response.json())
    assert response.status_code == 200

    response = client.post("/api/v1/collection/list")
    assert response.status_code == 200
    collections = response.json()
    assert 'test' in collections
    
    response = client.post("/api/v1/collection/delete", json={ "collection_name": "test" })
    assert response.status_code == 200 
    