"""Seed-derivation helpers for the unified LASER build pipeline."""

from __future__ import annotations

from collections import namedtuple

import numpy as np

SubSeeds = namedtuple("SubSeeds", ["pca", "medoid", "vamana", "rotator"])


def _spawn_one(child: np.random.SeedSequence) -> int:
    """Convert one SeedSequence child into a stable int32-compatible seed."""
    # faiss clustering seeds route through a signed C++ int; clamp to the
    # non-negative 31-bit range to avoid OverflowError in pybind.
    return int(child.generate_state(1, dtype=np.uint32)[0] & np.uint32(0x7FFFFFFF))


def derive_subseeds(master_seed: int) -> SubSeeds:
    """Derive four independent sub-seeds in fixed order.

    Order is always: ``[pca, medoid, vamana, rotator]``.
    """
    children = np.random.SeedSequence(int(master_seed)).spawn(4)
    return SubSeeds(*[_spawn_one(child) for child in children])
