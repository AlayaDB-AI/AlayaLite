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

"""Tests that Index.fit is deterministic across seeds."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from _laser_support import DISK_LASER_SUPPORTED  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DISK_LASER_SUPPORTED,
        reason="disk_laser is not supported on this build/platform",
    ),
]


def _vectors(n: int = 512, dim: int = 128, seed: int = 51) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _artifact_hashes(root: Path, name: str) -> dict[str, str]:
    suffixes = [
        "R64_MD128.index",
        "R64_MD128.index_rotator",
        "R64_MD128.index_cache_ids",
        "R64_MD128.index_cache_nodes",
    ]
    out = {}
    for suffix in suffixes:
        p = root / f"{name}_{suffix}"
        out[p.name] = _sha256(p)
    return out


def test_same_master_seed_is_deterministic_at_num_threads_1(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors()
    left = tmp_path / "left"
    right = tmp_path / "right"

    bp = laser.BuildParams(main_dim=128, R=64, disable_medoid=True)
    laser.Index.fit(
        vectors,
        output_dir=left,
        name="x",
        build_params=bp,
        num_threads=1,
        seed=42,
        auto_load=False,
    )
    laser.Index.fit(
        vectors,
        output_dir=right,
        name="x",
        build_params=bp,
        num_threads=1,
        seed=42,
        auto_load=False,
    )

    assert _artifact_hashes(left, "x") == _artifact_hashes(right, "x")


def test_different_master_seed_changes_artifacts(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors(seed=52)
    left = tmp_path / "left2"
    right = tmp_path / "right2"

    bp = laser.BuildParams(main_dim=128, R=64, disable_medoid=True)
    laser.Index.fit(
        vectors,
        output_dir=left,
        name="x",
        build_params=bp,
        num_threads=1,
        seed=42,
        auto_load=False,
    )
    laser.Index.fit(
        vectors,
        output_dir=right,
        name="x",
        build_params=bp,
        num_threads=1,
        seed=43,
        auto_load=False,
    )

    assert _artifact_hashes(left, "x") != _artifact_hashes(right, "x")
