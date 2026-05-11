# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Concurrent-search regression test for ``alayalite.DiskCollection(index_type="disk_laser")``.

Spec: ``openspec/changes/disk-laser-searcher-thread-safety``. Eight Python
threads issue the same query against one ``disk_laser`` collection and
their returned label lists are compared bit-exact against a single-thread
baseline captured before the threads start. The test enforces a 60 s
wall-clock budget so a deadlock surfaces as a failure rather than a hang.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from _laser_support import DISK_LASER_SUPPORTED  # noqa: E402
from alayalite import DiskCollection, MetricType

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="DiskCollection v1 is POSIX-only; disk_laser additionally requires Linux+libaio",
)

_SEG_PREFIX = "seg_00000001"
_FIXTURE_DIM = 128
_FIXTURE_R = 64

requires_laser = pytest.mark.skipif(
    not DISK_LASER_SUPPORTED,
    reason="disk_laser is not supported on this build/platform",
)


def _build_fixture(target_dir: Path, *, n: int = 256, seed: int = 1234):
    # pylint: disable=import-outside-toplevel
    from fixtures.laser.builder import build_small_laser_artifacts

    return build_small_laser_artifacts(
        target_dir,
        seg_basename=_SEG_PREFIX,
        n=n,
        dim=_FIXTURE_DIM,
        R=_FIXTURE_R,
        seed=seed,
    )


def _make_disk_laser(tmp_path: Path, name: str = "coll") -> DiskCollection:
    return DiskCollection(
        path=str(tmp_path / name),
        dim=_FIXTURE_DIM,
        metric=MetricType.L2,
        index_type="disk_laser",
    )


@requires_laser
def test_disk_laser_eight_threads_same_query_match_baseline(tmp_path):
    """Eight ThreadPoolExecutor workers, 100 calls each, all match baseline labels."""
    src_dir = tmp_path / "src"
    _, _, labels = _build_fixture(src_dir)
    col = _make_disk_laser(tmp_path)
    col.import_laser_segment(str(src_dir), labels)

    rng = np.random.default_rng(42)
    query = rng.standard_normal(_FIXTURE_DIM).astype(np.float32)

    search_kwargs = {"k": 10, "ef": 100, "beam_width": 4}
    baseline_hits = col.search(query, **search_kwargs)
    baseline_labels = [label for label, _ in baseline_hits]
    assert len(baseline_labels) == 10

    n_threads = 8
    iters_per_thread = 100
    start_event = threading.Event()

    def worker() -> list[bool]:
        start_event.wait()
        outcomes: list[bool] = []
        for _ in range(iters_per_thread):
            hits = col.search(query, **search_kwargs)
            outcomes.append([label for label, _ in hits] == baseline_labels)
        return outcomes

    deadline = time.monotonic() + 60.0
    # Manual lifecycle (no `with` context manager) so a deadline overrun does
    # not block on the implicit shutdown(wait=True) joining a stuck worker.
    # Note on deadlock recovery: cancel_futures=True only drops pending
    # submissions; threads that have already entered worker() cannot be killed
    # from Python. On a real deadlock pytest will surface the failure
    # immediately and the leaked workers exit with the test process, so the
    # CI runner does not hang.
    pool = ThreadPoolExecutor(max_workers=n_threads)
    try:
        futures = [pool.submit(worker) for _ in range(n_threads)]
        start_event.set()
        all_outcomes: list[bool] = []
        for future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail("concurrent disk_laser search exceeded 60 s wall-clock budget (possible deadlock)")
            all_outcomes.extend(future.result(timeout=remaining))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    assert len(all_outcomes) == n_threads * iters_per_thread
    failed = [i for i, ok in enumerate(all_outcomes) if not ok]
    assert not failed, (
        f"{len(failed)} of {len(all_outcomes)} concurrent search results diverged from the single-thread baseline"
    )
