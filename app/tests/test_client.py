import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_create_lists_delete_collection():
    response = client.post("/api/v1/collection/create", json={"collection_name": "test"})
    print(response.json())
    assert response.status_code == 200

    response = client.post("/api/v1/collection/list")
    assert response.status_code == 200
    collections = response.json()
    assert "test" in collections

    response = client.post("/api/v1/collection/delete", json={"collection_name": "test"})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_reset_collection():
    # insert collection
    response = client.post("/api/v1/collection/create", json={"collection_name": "test"})
    assert response.status_code == 200

    # reset collection
    response = client.post("/api/v1/collection/reset")
    assert response.status_code == 200

    response = client.post("/api/v1/collection/list")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_insert_collection():
    # insert collection
    client.post("/api/v1/collection/reset")
    response = client.post("/api/v1/collection/create", json={"collection_name": "test"})
    assert response.status_code == 200

    # insert items
    insert_payload = {
        "collection_name": "test",
        "items": [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]).tolist(), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]).tolist(), {"category": "B"}),
        ],
    }
    response = client.post("/api/v1/collection/insert", json=insert_payload)
    assert response.status_code == 200

    query_payload = {
        "collection_name": "test",
        "query_vector": [[0.1, 0.2, 0.3]],
        "limit": 2,
        "ef_search": 10,
        "num_threads": 1,
    }
    response = client.post("/api/v1/collection/query", json=query_payload)
    print(response.json())
    assert response.status_code == 200


async def test_upsert_collection():
    # insert collection
    client.post("/api/v1/collection/reset")
    response = client.post("/api/v1/collection/create", json={"collection_name": "test"})
    assert response.status_code == 200

    # insert items
    insert_payload = {
        "collection_name": "test",
        "items": [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]).tolist(), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]).tolist(), {"category": "B"}),
            (3, "Document 3", np.array([0.7, 0.8, 0.9]).tolist(), {"category": "C"}),
        ],
    }
    response = client.post("/api/v1/collection/insert", json=insert_payload)
    assert response.status_code == 200

    upsert_payload = {
        "collection_name": "test",
        "items": [
            (
                1,
                "New Document 1",
                np.array([0.1, 0.2, 0.3]).tolist(),
                {"category": "A"},
            ),
        ],
    }
    response = client.post("/api/v1/collection/upsert", json=upsert_payload)

    query_payload = {
        "collection_name": "test",
        "query_vector": [[0.1, 0.2, 0.3]],
        "limit": 2,
        "ef_search": 10,
        "num_threads": 1,
    }
    response = client.post("/api/v1/collection/query", json=query_payload)
    print(response.json())
    assert response.status_code == 200
