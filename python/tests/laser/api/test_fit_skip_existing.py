# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests that Index.fit skips segments whose files already exist."""

from __future__ import annotations

import time
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
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def test_skip_existing_skips_second_fit_and_false_rebuilds(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors(512, 128, 31)
    kwargs = {
        "output_dir": tmp_path,
        "name": "skip",
        "build_params": laser.BuildParams(main_dim=128, R=64, disable_medoid=True),
        "num_threads": 1,
        "seed": 42,
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


def test_skip_existing_rebuilds_when_seed_changes(tmp_path: Path) -> None:
    """``skip_existing=True`` SHALL rebuild when the seed sidecar disagrees.

    The per-artifact validators only check size + header, so without the
    sidecar gate ``fit(seed=Y, skip_existing=True)`` would silently reuse
    artifacts written earlier with ``seed=X`` and the caller would think
    they got a seed-Y index.
    """
    from alayalite import laser  # pylint: disable=import-outside-toplevel
    from alayalite.laser._idempotence import seed_sidecar_path  # pylint: disable=import-outside-toplevel

    vectors = _vectors(512, 128, 31)
    kwargs = {
        "output_dir": tmp_path,
        "name": "seed",
        "build_params": laser.BuildParams(main_dim=128, R=64, disable_medoid=True),
        "num_threads": 1,
        "skip_existing": True,
    }
    laser.Index.fit(vectors, seed=42, **kwargs)
    sidecar = Path(seed_sidecar_path(str(tmp_path / "seed")))
    assert sidecar.read_text(encoding="utf-8").strip() == "42"

    tracked = [
        tmp_path / "seed_pca_base.fbin",
        tmp_path / "seed_vamana_graph.index",
        tmp_path / "seed_R64_MD128.index",
    ]
    before = {p: p.stat().st_mtime_ns for p in tracked}

    time.sleep(1.1)
    laser.Index.fit(vectors, seed=43, **kwargs)
    after = {p: p.stat().st_mtime_ns for p in tracked}
    assert all(after[p] > before[p] for p in tracked)
    assert sidecar.read_text(encoding="utf-8").strip() == "43"


def test_skip_existing_rebuilds_when_seed_sidecar_missing(tmp_path: Path) -> None:
    """Legacy artifacts (no sidecar) SHALL force a rebuild even on seed match.

    A user with artifacts produced before this contract has no sidecar. The
    per-artifact validators cannot prove those artifacts came from the
    requested seed, so the conservative choice is to rebuild and republish
    the sidecar.
    """
    from alayalite import laser  # pylint: disable=import-outside-toplevel
    from alayalite.laser._idempotence import seed_sidecar_path  # pylint: disable=import-outside-toplevel

    vectors = _vectors(512, 128, 31)
    kwargs = {
        "output_dir": tmp_path,
        "name": "legacy",
        "build_params": laser.BuildParams(main_dim=128, R=64, disable_medoid=True),
        "num_threads": 1,
        "skip_existing": True,
    }
    laser.Index.fit(vectors, seed=42, **kwargs)
    sidecar = Path(seed_sidecar_path(str(tmp_path / "legacy")))
    sidecar.unlink()  # simulate pre-sidecar artifacts

    tracked = [
        tmp_path / "legacy_pca_base.fbin",
        tmp_path / "legacy_vamana_graph.index",
        tmp_path / "legacy_R64_MD128.index",
    ]
    before = {p: p.stat().st_mtime_ns for p in tracked}

    time.sleep(1.1)
    laser.Index.fit(vectors, seed=42, **kwargs)
    after = {p: p.stat().st_mtime_ns for p in tracked}
    assert all(after[p] > before[p] for p in tracked)
    assert sidecar.is_file()
    assert sidecar.read_text(encoding="utf-8").strip() == "42"


def test_skip_existing_rebuilds_pca_branch_when_main_dim_changes(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors(1024, 256, 32)
    base_kwargs = {
        "output_dir": tmp_path,
        "name": "skip_md",
        "num_threads": 1,
        "seed": 42,
        "skip_existing": True,
    }
    laser.Index.fit(
        vectors,
        build_params=laser.BuildParams(main_dim=128, R=64, disable_medoid=True),
        **base_kwargs,
    )
    pca_base = tmp_path / "skip_md_pca_base.fbin"
    first_mtime = pca_base.stat().st_mtime_ns
    assert (tmp_path / "skip_md_pca.bin").is_file()
    assert (tmp_path / "skip_md_R64_MD128.index").is_file()

    time.sleep(1.1)
    laser.Index.fit(
        vectors,
        build_params=laser.BuildParams(main_dim=256, R=64, disable_medoid=True),
        **base_kwargs,
    )
    second_mtime = pca_base.stat().st_mtime_ns

    # With a main-dim change, fit emits the new main index shape and removes
    # stale PCA params from the no-PCA branch.
    assert (tmp_path / "skip_md_R64_MD256.index").is_file()
    assert not (tmp_path / "skip_md_pca.bin").exists()
    assert second_mtime >= first_mtime
