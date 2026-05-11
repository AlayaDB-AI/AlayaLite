# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for Index.fit PCA and no-PCA code paths."""

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


def _vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def test_fit_runs_pca_when_main_dim_less_than_raw_dim(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors(n=1024, dim=256, seed=11)
    laser.Index.fit(
        vectors,
        output_dir=tmp_path,
        name="pca_on",
        build_params=laser.BuildParams(main_dim=128, R=64, disable_medoid=True),
        seed=42,
        num_threads=1,
    )

    assert (tmp_path / "pca_on_pca_base.fbin").is_file()
    assert (tmp_path / "pca_on_pca.bin").is_file()


def test_fit_skips_pca_when_main_dim_equals_raw_dim_and_warning_is_expected(tmp_path: Path, capfd) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors(n=512, dim=128, seed=12)
    laser.Index.fit(
        vectors,
        output_dir=tmp_path,
        name="pca_off",
        build_params=laser.BuildParams(main_dim=128, R=64, disable_medoid=True),
        seed=42,
        num_threads=1,
    )
    captured = capfd.readouterr()

    assert (tmp_path / "pca_off_pca_base.fbin").is_file()
    assert not (tmp_path / "pca_off_pca.bin").exists()
    assert "Warning: PCA file not found" in captured.err
