import os
import shutil

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


@pytest.mark.asyncio
async def test_persistence_across_restart(tmp_path, monkeypatch):
    storage_dir = str(tmp_path)
    monkeypatch.setenv("ALAYALITE_DATA_DIR", storage_dir)

    # Ensure we import a fresh app that will initialize Client with storage_dir
    import importlib
    import sys

    # Remove any previously loaded app.* modules so import picks up the env var
    for name in list(sys.modules.keys()):
        if name.startswith("app.") or name == "app":
            del sys.modules[name]

    app_module = importlib.import_module("app.main")
    test_app = app_module.app
    tc = TestClient(test_app)

    # create collection and insert an item
    resp = tc.post("/api/v1/collection/create", json={"collection_name": "restart_coll"})
    assert resp.status_code == 200

    insert_payload = {
        "collection_name": "restart_coll",
        "items": [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]).tolist(), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]).tolist(), {"category": "B"}),
        ],
    }
    resp = tc.post("/api/v1/collection/insert", json=insert_payload)
    assert resp.status_code == 200

    query_payload = {
        "collection_name": "restart_coll",
        "query_vector": [[0.1, 0.2, 0.3]],
        "limit": 2,
        "ef_search": 10,
        "num_threads": 1,
    }
    tc1_ans = tc.post("/api/v1/collection/query", json=query_payload)
    assert tc1_ans.status_code == 200

    # save collection to disk
    resp = tc.post("/api/v1/collection/save", json={"collection_name": "restart_coll"})
    assert resp.status_code == 200

    # verify files exist on disk
    coll_path = os.path.join(storage_dir, "restart_coll")
    assert os.path.isdir(coll_path)
    assert os.path.isfile(os.path.join(coll_path, "schema.json"))

    # Simulate restart: force reload of app modules and create a new TestClient
    for name in list(sys.modules.keys()):
        if name.startswith("app.") or name == "app":
            del sys.modules[name]

    app_module2 = importlib.import_module("app.main")
    test_app2 = app_module2.app
    tc2 = TestClient(test_app2)

    # list collections should include our saved collection
    resp = tc2.post("/api/v1/collection/list")
    assert resp.status_code == 200
    assert "restart_coll" in resp.json()

    tc2_ans = tc2.post("/api/v1/collection/query", json=query_payload)
    assert tc2_ans.status_code == 200
    assert tc1_ans.json() == tc2_ans.json()

    # cleanup
    try:
        shutil.rmtree(coll_path)
    except Exception:
        pass
