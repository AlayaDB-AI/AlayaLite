import importlib
import sys
from typing import AsyncGenerator

import httpx
import pytest_asyncio


def _reload_app_module():
    # Remove loaded app.* modules so a fresh app is created with current env
    for name in list(sys.modules.keys()):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]
    app_module = importlib.import_module("app.main")
    return app_module.app


@pytest_asyncio.fixture()
async def fresh_client(tmp_path, monkeypatch) -> AsyncGenerator[httpx.AsyncClient, None]:
    # Isolate storage into a temp directory for this test
    monkeypatch.setenv("ALAYALITE_DATA_DIR", str(tmp_path))
    # Set RocksDB directory to tmp_path for test isolation
    rocksdb_dir = str(tmp_path / "RocksDB")
    monkeypatch.setenv("ALAYALITE_ROCKSDB_DIR", rocksdb_dir)
    app = _reload_app_module()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
