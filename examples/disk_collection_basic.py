# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Minimal end-to-end demo of alayalite.DiskCollection.

Builds a small disk-resident segment, flushes it, closes the handle,
reopens via the static factory, and runs both an exact-match query and a
top-k brute-force search. Mirrors the proposal's minimal example from
openspec/changes/add-disk-collection-flat/proposal.md.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from alayalite import DiskCollection, MetricType


def main() -> None:
    rng = np.random.default_rng(2026)
    n_rows = 1000
    dim = 32

    # Generate a random dataset and a random query.
    vectors = rng.standard_normal((n_rows, dim)).astype(np.float32)
    ids = np.arange(n_rows, dtype=np.uint64)
    query = rng.standard_normal(dim).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "demo_collection")

        # ---- Phase 1: create and populate -------------------------------
        col = DiskCollection(
            path=path,
            dim=dim,
            metric=MetricType.L2,
            index_type="disk_flat",
        )
        col.add(vectors, ids)
        col.flush()
        print(f"after flush: size={col.size()}, dim={col.dim()}")

        # Search before close (should match brute force).
        hits = col.search(query, k=5)
        print(f"top-5 (pre-close): {hits}")

        # Drop the handle to simulate a process restart.
        del col

        # ---- Phase 2: reopen and re-search ------------------------------
        reopened = DiskCollection.open(path)
        assert reopened.size() == n_rows
        hits_again = reopened.search(query, k=5)
        print(f"top-5 (reopen):    {hits_again}")
        assert hits == hits_again, "results must be identical across reopen"

        # ---- Phase 3: exact-match sanity --------------------------------
        # An L2 query that equals a stored vector returns distance 0.0.
        exact_hits = reopened.search(vectors[42], k=1)
        assert exact_hits[0] == (42, 0.0), f"unexpected: {exact_hits[0]}"
        print(f"exact-match hit:   {exact_hits[0]}")

    print("OK")


if __name__ == "__main__":
    main()
