# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for DiskCollection's process-level collection lock."""

import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pytest
from alayalite import DiskCollection, MetricType

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="DiskCollection v1 is POSIX-only")

LOCK_BUSY = "collection is already open by another process"


def _vectors(n, dim):
    values = np.arange(n * dim, dtype=np.float32).reshape(n, dim)
    return values + np.float32(1.0)


def _ids(n, base=1000):
    return np.arange(base, base + n, dtype=np.uint64)


def _build_collection(path):
    col = DiskCollection(str(path), 4, MetricType.L2, "disk_flat")
    col.add(_vectors(2, 4), _ids(2))
    col.flush()
    del col


def _lock_path_text(path):
    return str((pathlib.Path(path) / ".lock").resolve())


def _open_worker(path, queue):
    try:
        DiskCollection.open(str(path))
    # pylint: disable=broad-exception-caught
    except BaseException as exc:  # noqa: BLE001 - subprocess must serialize all failures.
        queue.put((type(exc).__name__, str(exc)))
        return
    queue.put(("NO_ERROR", ""))


def test_double_open_same_process_raises_runtime_error(tmp_path):
    path = tmp_path / "coll"
    col = DiskCollection(str(path), 4, MetricType.L2, "disk_flat")

    with pytest.raises(RuntimeError) as exc_info:
        DiskCollection.open(str(path))

    msg = str(exc_info.value)
    assert _lock_path_text(path) in msg
    assert LOCK_BUSY in msg
    del col


def test_double_open_across_subprocess_raises_runtime_error(tmp_path):
    # Use mp.get_context("fork") only; do NOT call mp.set_start_method which
    # mutates process-global state and can flake other tests in the session.
    ctx = mp.get_context("fork")
    path = tmp_path / "coll"
    col = DiskCollection(str(path), 4, MetricType.L2, "disk_flat")
    queue = ctx.Queue()
    proc = ctx.Process(target=_open_worker, args=(path, queue))

    proc.start()
    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)

    assert proc.exitcode == 0
    kind, msg = queue.get(timeout=1)
    assert kind == "RuntimeError", msg
    assert _lock_path_text(path) in msg
    assert LOCK_BUSY in msg
    del col


def test_legacy_collection_without_lock_opens_cleanly(tmp_path):
    path = tmp_path / "coll"
    _build_collection(path)
    (path / ".lock").unlink(missing_ok=True)

    col = DiskCollection.open(str(path))

    assert col.size() == 2
    assert (path / ".lock").is_file()
