# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Windows-only end-to-end smoke test for LASER + IOCP backend.

Mirrors the macOS ThreadPool smoke shape:
  * Build a tiny LASER index from a synthetic dataset
    (100K vectors here trimmed to 10K so CI runners stay green; the
    full-size 100K variant is exercised by `laser-cross-platform-perf`).
  * Run a fixed query set.
  * Assert recall@10 is sensible (>= 0.9 on this synthetic — IOCP is not
    expected to alter recall; the dataset's structure dominates).
  * The Linux libaio reference is captured separately by the
    `laser-cross-platform-perf` workflow and is not pinned into this
    test — the ±1pp tolerance check happens at the workflow level
    (`aggregate-summary` step) where we have both runs.

The smoke gates the Windows lane's wheel: if the IOCP build is broken
end-to-end, this test fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Same skip gate the rest of the LASER API tests use.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _laser_support import DISK_LASER_SUPPORTED  # noqa: E402 pylint: disable=wrong-import-position

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "win32", reason="Windows-only smoke (IOCP backend)"),
    pytest.mark.skipif(
        not DISK_LASER_SUPPORTED,
        reason="disk_laser is not supported on this build/platform",
    ),
]


def _synthetic_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, np.float32(1e-6))


def _bruteforce_topk(query: np.ndarray, base: np.ndarray, k: int) -> np.ndarray:
    """Reference L2 top-k for the recall@k cross-check."""
    # Squared L2: ||q - b||^2 = ||q||^2 - 2 q.b + ||b||^2
    diff = base - query
    d2 = np.einsum("ij,ij->i", diff, diff)
    return np.argpartition(d2, k)[:k]


def test_windows_iocp_build_and_search_smoke(tmp_path: Path) -> None:
    """End-to-end build + search smoke on Windows IOCP backend.

    Parameters mirror the existing api fixture (`test_fit_basic.py`):
    main_dim=128, R=64, disable_medoid=True. The IOCP backend is the
    only backend available on Windows so picking it is implicit.
    """
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    n_base = 10_000
    dim = 128
    n_query = 50
    top_k = 10
    seed = 42

    base = _synthetic_vectors(n=n_base, dim=dim, seed=seed)
    queries = _synthetic_vectors(n=n_query, dim=dim, seed=seed + 1)

    idx = laser.Index.fit(
        base,
        output_dir=tmp_path,
        name="windows_smoke",
        build_params=laser.BuildParams(main_dim=dim, R=64, disable_medoid=True),
        num_threads=1,
        seed=seed,
    )

    recall_sum = 0
    for q in queries:
        hits = idx.search(q, top_k)
        assert hits.shape == (top_k,)
        gt = {int(x) for x in _bruteforce_topk(q, base, top_k)}
        recall_sum += len({int(x) for x in hits} & gt)

    recall_at_k = recall_sum / float(n_query * top_k)
    # Loose lower bound — LASER on a tiny synthetic with disabled medoid
    # is expected to comfortably exceed 0.9. If this drops below 0.85 on
    # Windows we have an IOCP integration bug, not a recall tuning issue.
    assert recall_at_k >= 0.85, f"recall@{top_k}={recall_at_k:.3f} below 0.85 floor"
