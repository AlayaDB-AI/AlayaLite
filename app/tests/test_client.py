import importlib
import os
import shutil
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient


def reload_client():
    # Simulate restart: force reload of app modules and create a new TestClient
    for name in list(sys.modules.keys()):
        if name.startswith("app.") or name == "app":
            del sys.modules[name]

    app_module = importlib.import_module("app.main")
    test_app = app_module.app
    tc = TestClient(test_app)

    return tc


@pytest.mark.asyncio
async def test_create_lists_delete_collection(fresh_client: TestClient):
    client = fresh_client
    response = client.post("/api/v1/collection/create", json={"collection_name": "test"})
    print(response.json())
    assert response.status_code == 200

    response = client.post("/api/v1/collection/create", json={"collection_name": "test"})
    assert response.status_code == 409

    response = client.post("/api/v1/collection/list")
    assert response.status_code == 200
    collections = response.json()
    assert "test" in collections

    response = client.post("/api/v1/collection/delete", json={"collection_name": "test"})
    assert response.status_code == 200

    response = client.post("/api/v1/collection/delete", json={"collection_name": "test"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_reset_collection(fresh_client: TestClient):
    client = fresh_client
    # insert collection
    response = client.post("/api/v1/collection/create", json={"collection_name": "test"})
    assert response.status_code == 200

    # reset collection
    response = client.post("/api/v1/collection/reset", json={"delete_on_disk": False})
    assert response.status_code == 200

    response = client.post("/api/v1/collection/list")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_reset_persistence_collection():
    tc = reload_client()
    collection_name_list = ["a", "b", "c", "d", "e"]
    for collection_name in collection_name_list:
        response = tc.post("/api/v1/collection/create", json={"collection_name": collection_name})
        assert response.status_code == 200

        response = tc.post(
            "/api/v1/collection/insert",
            json={
                "collection_name": collection_name,
                "items": [
                    (1, "Document 1", np.array([0.1, 0.2, 0.3]).tolist(), {"category": "A"}),
                ],
            },
        )

        response = tc.post("/api/v1/collection/save", json={"collection_name": collection_name})
        assert response.status_code == 200

    response = tc.post("/api/v1/collection/list")
    assert response.status_code == 200
    assert set(response.json()) == set(collection_name_list)

    # reset collection
    response = tc.post("/api/v1/collection/reset", json={"delete_on_disk": True})
    assert response.status_code == 200

    tc2 = reload_client()
    response = tc2.post("/api/v1/collection/list")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_insert_collection(fresh_client: TestClient):
    client = fresh_client
    # insert collection
    client.post("/api/v1/collection/reset")

    # insert items
    insert_payload = {
        "collection_name": "test",
        "items": [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]).tolist(), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]).tolist(), {"category": "B"}),
        ],
    }

    # Inserting into a non-existent collection should return 404
    response = client.post("/api/v1/collection/insert", json=insert_payload)
    assert response.status_code == 404

    # create collection and insert
    response = client.post("/api/v1/collection/create", json={"collection_name": "test"})
    assert response.status_code == 200
    response = client.post("/api/v1/collection/insert", json=insert_payload)
    assert response.status_code == 200

    # if the items not have same length, should return 422
    bad_insert_payload = {
        "collection_name": "test",
        "items": [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]).tolist()),  # Missing metadata
            (2, "Document 2", np.array([0.4, 0.5, 0.6]).tolist(), {"category": "B"}),
        ],
    }
    response = client.post("/api/v1/collection/insert", json=bad_insert_payload)
    assert response.status_code == 422

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
async def test_upsert_collection(fresh_client: TestClient):
    client = fresh_client
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

    # upsert into a non-existent collection should return 404
    upsert_payload = {
        "collection_name": "nope",
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
    assert response.status_code == 404

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
async def test_query_collection(fresh_client: TestClient):
    client = fresh_client
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

    # limit higher than collection size
    query_payload = {
        "collection_name": "test",
        "query_vector": [[0.1, 0.2, 0.3]],
        "limit": 11,
        "ef_search": 10,
        "num_threads": 1,
    }
    response = client.post("/api/v1/collection/query", json=query_payload)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_persistence_across_restart(tmp_path, monkeypatch):
    storage_dir = str(tmp_path)
    monkeypatch.setenv("ALAYALITE_DATA_DIR", storage_dir)

    tc = reload_client()

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

    tc2 = reload_client()
    # list collections should include our saved collection
    resp = tc2.post("/api/v1/collection/list")
    assert resp.status_code == 200
    assert "restart_coll" in resp.json()

    tc2_ans = tc2.post("/api/v1/collection/query", json=query_payload)
    assert tc2_ans.status_code == 200
    assert tc1_ans.json() == tc2_ans.json()

    # delete on disk
    resp = tc2.post("/api/v1/collection/delete", json={"collection_name": "restart_coll", "delete_on_disk": True})
    assert resp.status_code == 200

    # list collections should include our saved collection
    resp = tc2.post("/api/v1/collection/list")
    assert resp.status_code == 200
    assert "restart_coll" not in resp.json()

    tc3 = reload_client()
    # list collections should include our saved collection
    resp = tc3.post("/api/v1/collection/list")
    assert resp.status_code == 200
    assert "restart_coll" not in resp.json()

    # cleanup
    try:
        shutil.rmtree(coll_path)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_operations_on_nonexistent_collection(fresh_client: TestClient):
    client = fresh_client
    # query on a non-existent collection should raise error
    payload = {
        "collection_name": "nope",
        "query_vector": [[0.0, 0.0, 0.0]],
        "limit": 1,
        "ef_search": 10,
        "num_threads": 1,
    }
    resp = client.post("/api/v1/collection/query", json=payload)
    assert resp.status_code == 404
    assert "error" in resp.json()

    # delete_by_id on non-existent collection
    resp = client.post("/api/v1/collection/delete_by_id", json={"collection_name": "nope", "ids": [1]})
    assert resp.status_code == 404
    assert "error" in resp.json()

    # delete_by_filter on non-existent collection
    resp = client.post(
        "/api/v1/collection/delete_by_filter",
        json={"collection_name": "nope", "filter": {"k": "v"}},
    )
    assert resp.status_code == 404
    assert "error" in resp.json()


@pytest.mark.asyncio
async def test_delete_by_id_and_filter(fresh_client: TestClient):
    client = fresh_client
    client.post("/api/v1/collection/reset")
    resp = client.post("/api/v1/collection/create", json={"collection_name": "test"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_duplicate_collection_creation_conflict(fresh_client: TestClient):
    client = fresh_client
    resp = client.post("/api/v1/collection/create", json={"collection_name": "dup"})
    assert resp.status_code == 200
    # Second creation should return 409 conflict with error message
    resp2 = client.post("/api/v1/collection/create", json={"collection_name": "dup"})
    assert resp2.status_code == 409
    body = resp2.json()
    assert "error" in body and "already exists" in body["error"]


def test_root_endpoint(fresh_client: TestClient):
    client = fresh_client
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data and "AlayaLite" in data["message"]

    # create collection
    resp = client.post("/api/v1/collection/create", json={"collection_name": "test"})
    assert resp.status_code == 200

    insert_payload = {
        "collection_name": "test",
        "items": [
            (1, "Document 1", np.array([0.1, 0.2, 0.3]).tolist(), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6]).tolist(), {"category": "B"}),
            (3, "Document 3", np.array([0.7, 0.8, 0.9]).tolist(), {"category": "A"}),
        ],
    }
    client.post("/api/v1/collection/insert", json=insert_payload)
    assert resp.status_code == 200

    # delete by id
    resp = client.post("/api/v1/collection/delete_by_id", json={"collection_name": "test", "ids": ["2"]})
    assert resp.status_code == 200

    # delete by filter
    resp = client.post(
        "/api/v1/collection/delete_by_filter",
        json={"collection_name": "test", "filter": {"category": "A"}},
    )
    assert resp.status_code == 200
