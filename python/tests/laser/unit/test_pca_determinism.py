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

"""Determinism checks for the Python LASER PCA path."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

threadpoolctl = pytest.importorskip("threadpoolctl")
pytest.importorskip("sklearn")


def _write_fbin(path: Path, vectors: np.ndarray) -> None:
    n, dim = vectors.shape
    with path.open("wb") as f:
        np.array([n, dim], dtype=np.int32).tofile(f)
        vectors.astype(np.float32, copy=False).tofile(f)


def _run_pca_once(base_path: Path, out_dir: Path) -> tuple[bytes, bytes]:
    from alayalite.laser._pca import (  # pylint: disable=import-outside-toplevel
        fit_incremental_pca,
        pca_transform_and_save,
        sample_vectors_from_fbin,
        save_pca_params,
    )

    out_dir.mkdir()
    vectors, sample_vectors = sample_vectors_from_fbin(
        str(base_path),
        sample_ratio=0.5,
        seed=777,
    )
    pca = fit_incremental_pca(sample_vectors, n_components=vectors.shape[1], batch_size=40)
    pca_path = out_dir / "dsqg_test_pca.bin"
    pca_base_path = out_dir / "dsqg_test_pca_base.fbin"
    save_pca_params(pca, str(pca_path))
    pca_transform_and_save(vectors, pca, str(pca_base_path), chunk_size=37)
    return pca_path.read_bytes(), pca_base_path.read_bytes()


def test_pca_outputs_are_byte_identical_with_fixed_seed_and_single_thread(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    vectors = rng.normal(size=(160, 16)).astype(np.float32)
    vectors += np.linspace(0.0, 0.5, num=16, dtype=np.float32)
    base_path = tmp_path / "base.fbin"
    _write_fbin(base_path, vectors)

    with threadpoolctl.threadpool_limits(limits=1):
        first_pca, first_pca_base = _run_pca_once(base_path, tmp_path / "first")
        second_pca, second_pca_base = _run_pca_once(base_path, tmp_path / "second")

    assert first_pca == second_pca
    assert first_pca_base == second_pca_base


def test_sample_vectors_keeps_at_least_raw_dim_rows_when_available(tmp_path: Path) -> None:
    from alayalite.laser._pca import sample_vectors_from_fbin  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(2025)
    vectors = rng.normal(size=(512, 256)).astype(np.float32)
    base_path = tmp_path / "base.fbin"
    _write_fbin(base_path, vectors)

    _, sample_vectors = sample_vectors_from_fbin(
        str(base_path),
        sample_ratio=0.25,
        seed=123,
    )

    assert sample_vectors.shape == (256, 256)


def test_sample_vectors_returns_all_rows_when_raw_dim_exceeds_count(tmp_path: Path) -> None:
    from alayalite.laser._pca import sample_vectors_from_fbin  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(2026)
    vectors = rng.normal(size=(128, 256)).astype(np.float32)
    base_path = tmp_path / "small_base.fbin"
    _write_fbin(base_path, vectors)

    _, sample_vectors = sample_vectors_from_fbin(
        str(base_path),
        sample_ratio=0.25,
        seed=123,
    )

    assert sample_vectors.shape == vectors.shape


def test_fit_incremental_pca_rejects_too_few_samples_for_components() -> None:
    from alayalite.laser._pca import fit_incremental_pca  # pylint: disable=import-outside-toplevel

    sample_vectors = np.zeros((128, 256), dtype=np.float32)

    with pytest.raises(ValueError, match="at least n_components"):
        fit_incremental_pca(sample_vectors, n_components=256)


def test_force_single_thread_requires_all_alignment_seeds(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "laser_example_main",
        repo_root / "examples" / "laser" / "main.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = tmp_path / "missing_pca_seed.toml"
    cfg.write_text(
        """
medoid_seed = 42
rotator_seed = 42
force_single_thread = true
build_threads = 1

[dataset]
name = "test"
metric = "l2"
degree = 64
main_dimension = 16

[paths]
base = "/tmp/base.fbin"
query = "/tmp/query.fbin"
gt = "/tmp/gt.ibin"
output = "/tmp/out"
""".lstrip(),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        topk=None,
        threads=None,
        beam_width=None,
        dram_budget=None,
        efs=None,
        ep_num=None,
        degree=None,
        main_dim=None,
        build_threads=None,
        ef_indexing=None,
        warmup=None,
        runs=None,
    )

    with pytest.raises(ValueError, match="pca_seed"):
        module.load_config(cfg, args)
