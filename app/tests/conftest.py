import importlib
import sys
from typing import Generator

import pytest
from fastapi.testclient import TestClient


def _reload_app_module() -> TestClient:
    # Remove loaded app.* modules so a fresh client is created with current env
    for name in list(sys.modules.keys()):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]
    app_module = importlib.import_module("app.main")
    return TestClient(app_module.app)


@pytest.fixture()
def fresh_client(tmp_path, monkeypatch) -> Generator[TestClient, None, None]:
    # Isolate storage into a temp directory for this test
    monkeypatch.setenv("ALAYALITE_DATA_DIR", str(tmp_path))
    # Set RocksDB directory to tmp_path for test isolation
    rocksdb_dir = str(tmp_path / "RocksDB")
    monkeypatch.setenv("ALAYALITE_ROCKSDB_DIR", rocksdb_dir)
    client = _reload_app_module()
    try:
        yield client
    finally:
        # tmp_path will be cleaned up by pytest automatically
        pass
