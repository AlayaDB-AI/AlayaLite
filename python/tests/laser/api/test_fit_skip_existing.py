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

"""Tests that Index.fit skips segments whose files already exist."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from _laser_support import DISK_LASER_SUPPORTED  # noqa: E402

pytestmark = pytest.mark.skipif(
    not DISK_LASER_SUPPORTED,
    reason="disk_laser is not supported on this build/platform",
)


def _vectors(n: int, dim: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def test_skip_existing_skips_second_fit_and_false_rebuilds(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors(512, 128, 31)
    kwargs = {
        "output_dir": tmp_path,
        "name": "skip",
        "main_dim": 128,
        "R": 64,
        "num_threads": 1,
        "seed": 42,
        "disable_medoid": True,
    }
    laser.Index.fit(vectors, skip_existing=True, **kwargs)

    tracked = [
        tmp_path / "skip_pca_base.fbin",
        tmp_path / "skip_vamana_graph.index",
        tmp_path / "skip_R64_MD128.index",
    ]
    before = {p: p.stat().st_mtime_ns for p in tracked}

    time.sleep(1.1)
    laser.Index.fit(vectors, skip_existing=True, **kwargs)
    same = {p: p.stat().st_mtime_ns for p in tracked}
    assert before == same

    time.sleep(1.1)
    laser.Index.fit(vectors, skip_existing=False, **kwargs)
    rebuilt = {p: p.stat().st_mtime_ns for p in tracked}
    assert any(rebuilt[p] > same[p] for p in tracked)


def test_skip_existing_rebuilds_pca_branch_when_main_dim_changes(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors(1024, 256, 32)
    base_kwargs = {
        "output_dir": tmp_path,
        "name": "skip_md",
        "R": 64,
        "num_threads": 1,
        "seed": 42,
        "disable_medoid": True,
        "skip_existing": True,
    }
    laser.Index.fit(vectors, main_dim=128, **base_kwargs)
    pca_base = tmp_path / "skip_md_pca_base.fbin"
    first_mtime = pca_base.stat().st_mtime_ns
    assert (tmp_path / "skip_md_pca.bin").is_file()
    assert (tmp_path / "skip_md_R64_MD128.index").is_file()

    time.sleep(1.1)
    laser.Index.fit(vectors, main_dim=256, **base_kwargs)
    second_mtime = pca_base.stat().st_mtime_ns

    # With a main-dim change, fit emits the new main index shape and removes
    # stale PCA params from the no-PCA branch.
    assert (tmp_path / "skip_md_R64_MD256.index").is_file()
    assert not (tmp_path / "skip_md_pca.bin").exists()
    assert second_mtime >= first_mtime
