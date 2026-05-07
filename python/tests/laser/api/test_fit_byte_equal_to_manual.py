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

"""Tests that Index.fit output is byte-equal to a manually assembled index."""

from __future__ import annotations

import hashlib
import shutil
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


def _vectors(n: int = 512, dim: int = 128, seed: int = 61) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def _write_fbin(path: Path, vectors: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array([vectors.shape[0], vectors.shape[1]], dtype=np.int32).tofile(f)
        vectors.astype(np.float32, copy=False).tofile(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _hashes(root: Path, name: str) -> dict[str, str]:
    suffixes = [
        "R64_MD128.index",
        "R64_MD128.index_rotator",
        "R64_MD128.index_cache_ids",
        "R64_MD128.index_cache_nodes",
    ]
    return {suffix: _sha256(root / f"{name}_{suffix}") for suffix in suffixes}


def _build_manual_pipeline(root: Path, name: str, vectors: np.ndarray, *, seed: int) -> None:
    # pylint: disable=import-outside-toplevel
    from alayalite import vamana
    from alayalite._alayalitepy import laser as raw_laser

    root.mkdir(parents=True, exist_ok=True)
    prefix = root / name
    raw_path = root / f"{name}_raw.fbin"
    pca_base = root / f"{name}_pca_base.fbin"
    graph = root / f"{name}_vamana_graph.index"

    _write_fbin(raw_path, vectors)
    shutil.copyfile(raw_path, pca_base)
    vamana.build_index(
        data_path=str(raw_path),
        output_path=str(graph),
        R=64,
        L=200,
        alpha=1.2,
        seed=seed,
        num_threads=1,
        dram_budget_gb=1.0,
    )

    idx = raw_laser.Index(
        index_type="QG",
        metric="l2",
        num_elements=vectors.shape[0],
        main_dimension=128,
        dimension=128,
        degree_bound=64,
        rotator_seed=seed,
        rotator_dump_path="",
    )
    idx.build_index(str(graph), str(prefix), EF=200, num_thread=1)


def test_unified_fit_is_byte_equal_to_manual_pipeline_at_single_thread(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors()
    manual = tmp_path / "manual"
    unified = tmp_path / "unified"

    _build_manual_pipeline(manual, "x", vectors, seed=42)
    laser.Index.fit(
        vectors,
        output_dir=unified,
        name="x",
        build_params=laser.BuildParams(
            main_dim=128,
            R=64,
            L=200,
            alpha=1.2,
            ef_indexing=200,
            disable_medoid=True,
        ),
        num_threads=1,
        seed=42,
        auto_load=False,
        skip_existing=False,
    )

    assert _hashes(manual, "x") == _hashes(unified, "x")
