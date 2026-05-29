import gc
import importlib
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx


def _close_app_client() -> None:
    router_module = sys.modules.get("app.routers.client")
    app_client = getattr(router_module, "client", None)
    if app_client is None:
        return

    try:
        app_client.reset(delete_on_disk=False)
    except (AttributeError, RuntimeError):
        pass


def _clear_test_modules() -> None:
    _close_app_client()
    gc.collect()
    for name in list(sys.modules.keys()):
        if (
            name == "app.main"
            or name == "app.routers"
            or name.startswith("app.routers.")
            or name.startswith("alayalite")
        ):
            del sys.modules[name]
    gc.collect()


@asynccontextmanager
async def create_app_client() -> AsyncIterator[httpx.AsyncClient]:
    _clear_test_modules()
    app_module = importlib.import_module("app.main")
    transport = httpx.ASGITransport(app=app_module.app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client
    finally:
        _clear_test_modules()
