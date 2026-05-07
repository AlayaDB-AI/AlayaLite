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

"""Tests for Index.fit medoid-initialisation path."""

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


def _vectors(n: int = 512, dim: int = 128, seed: int = 21) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def test_fit_writes_medoids_by_default(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors()
    laser.Index.fit(
        vectors,
        output_dir=tmp_path,
        name="medoid_on",
        build_params=laser.BuildParams(main_dim=128, R=64, ep_num=16),
        num_threads=1,
        seed=42,
    )

    assert (tmp_path / "medoid_on_medoids").is_file()
    assert (tmp_path / "medoid_on_medoids_indices").is_file()


def test_fit_disable_medoid_skips_files_but_search_still_works(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors(seed=22)
    idx = laser.Index.fit(
        vectors,
        output_dir=tmp_path,
        name="medoid_off",
        build_params=laser.BuildParams(main_dim=128, R=64, ep_num=16, disable_medoid=True),
        num_threads=1,
        seed=42,
    )

    assert not (tmp_path / "medoid_off_medoids").exists()
    assert not (tmp_path / "medoid_off_medoids_indices").exists()
    hits = idx.search(vectors[0], 10)
    assert hits.shape == (10,)
