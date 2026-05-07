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


# Phase 4 task 4.5: dual-substring error literals SHALL propagate from C++
# `std::runtime_error` to Python `RuntimeError` verbatim. The Python binding
# delegates to pybind11's default exception translator for these (no
# special-case mapping), so we assert the propagation contract directly.
def test_open_collection_in_progress_propagates_literal(tmp_path):
    path = tmp_path / "coll"
    path.mkdir()  # mimic ctor mid-flight: dir exists, no .lock, no manifest

    with pytest.raises(RuntimeError) as exc_info:
        DiskCollection.open(str(path))

    msg = str(exc_info.value)
    assert "target path is a collection-in-progress, not yet published" in msg, msg
    assert _lock_path_text(path) in msg, msg


def _ctor_worker(path, queue):
    try:
        col = DiskCollection(str(path), 4, MetricType.L2, "disk_flat")
        queue.put(("OK", ""))
        del col
        return
    # pylint: disable=broad-exception-caught
    except BaseException as exc:  # noqa: BLE001 - serialize across process boundary
        queue.put((type(exc).__name__, str(exc)))


def test_two_concurrent_ctors_one_wins_other_gets_runtime_error(tmp_path):
    """Two concurrent ctors against an absent path: exactly one SHALL succeed; the other
    SHALL raise RuntimeError whose message contains a documented dual substring (the
    weakly_canonical .lock path AND either "target path already exists" or
    "is being created concurrently"). The on-disk state SHALL be a complete collection.
    """
    ctx = mp.get_context("fork")
    path = tmp_path / "race_coll"
    queue1 = ctx.Queue()
    queue2 = ctx.Queue()
    p1 = ctx.Process(target=_ctor_worker, args=(path, queue1))
    p2 = ctx.Process(target=_ctor_worker, args=(path, queue2))
    p1.start()
    p2.start()
    p1.join(timeout=10)
    p2.join(timeout=10)
    if p1.is_alive():
        p1.terminate()
    if p2.is_alive():
        p2.terminate()
    assert p1.exitcode == 0 and p2.exitcode == 0

    r1 = queue1.get(timeout=1)
    r2 = queue2.get(timeout=1)
    successes = [r for r in (r1, r2) if r[0] == "OK"]
    failures = [r for r in (r1, r2) if r[0] != "OK"]
    assert len(successes) == 1, (r1, r2)
    assert len(failures) == 1, (r1, r2)

    kind, msg = failures[0]
    assert kind == "RuntimeError", (kind, msg)
    assert _lock_path_text(path) in msg or path.name in msg, msg
    assert "target path already exists" in msg or "is being created concurrently" in msg, msg

    # On-disk state: full collection.
    assert (path / ".lock").is_file()
    assert (path / "segments").is_dir()
    assert (path / "collection_manifest.txt").is_file()
