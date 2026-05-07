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

"""Tests for Index.fit basic functionality."""

from __future__ import annotations

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


def _vectors(n: int = 512, dim: int = 128, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x /= np.maximum(norms, np.float32(1e-6))
    return x


def _write_fbin(path: Path, vectors: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array([vectors.shape[0], vectors.shape[1]], dtype=np.int32).tofile(f)
        vectors.astype(np.float32, copy=False).tofile(f)


def test_fit_autoload_search_with_ndarray_input(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors()
    idx = laser.Index.fit(
        vectors,
        output_dir=tmp_path,
        name="u_nd",
        build_params=laser.BuildParams(main_dim=128, R=64, disable_medoid=True),
        num_threads=1,
        seed=42,
    )
    hits = idx.search(vectors[0], 10)
    assert hits.shape == (10,)
    assert hits.dtype == np.uint32


def test_fit_autoload_search_with_fbin_input(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors()
    raw = tmp_path / "base.fbin"
    _write_fbin(raw, vectors)

    idx = laser.Index.fit(
        str(raw),
        output_dir=tmp_path,
        name="u_fbin",
        build_params=laser.BuildParams(main_dim=128, R=64, disable_medoid=True),
        num_threads=1,
        seed=42,
    )
    hits = idx.search(vectors[1], 10)
    assert hits.shape == (10,)
    # fbin input path is canonical raw input; fit must not rewrite *_raw.fbin.
    assert not (tmp_path / "u_fbin_raw.fbin").exists()
