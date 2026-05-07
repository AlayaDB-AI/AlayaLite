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

"""Build small LASER native artifacts for `import_laser_segment` tests.

The builder produces the four LASER artifacts (`*_R<R>_MD<dim>.index`,
`_rotator`, `_cache_ids`, `_cache_nodes`) that match the C++
`LaserSegmentImporter`'s naming contract:

    dsqg_<seg_basename>_R<R>_MD<main_dim>.index
    dsqg_<seg_basename>_R<R>_MD<main_dim>.index_rotator
    dsqg_<seg_basename>_R<R>_MD<main_dim>.index_cache_ids
    dsqg_<seg_basename>_R<R>_MD<main_dim>.index_cache_nodes

The function is lazily imported inside test bodies to keep the
unsupported-build wheel matrix green: `alayalite.laser` and
`alayalite.vamana` are not loadable on Linux+OFF / macOS / Windows.
Tests SHALL gate the call on the `_laser_support.DISK_LASER_SUPPORTED`
runtime probe so the builder is never invoked on an unsupported build.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _generate_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    """Stable random vectors with bounded dynamic range."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(loc=0.0, scale=1.0, size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, np.float32(1e-6))
    return vectors


def build_small_laser_artifacts(
    target_dir: Path,
    *,
    seg_basename: str = "seg_00000001",
    n: int = 256,
    dim: int = 128,
    R: int = 64,
    seed: int = 1234,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Build a deterministic LASER fixture in `target_dir`.

    Parameters
    ----------
    target_dir : Path
        Directory in which the four required LASER artifacts (and the
        intermediate Vamana / data files) are written. Created if missing.
    seg_basename : str
        Used for the `dsqg_<seg_basename>_*` filename prefix the C++
        `LaserSegmentImporter` expects.
    n : int
        Number of points (rows) in the LASER index. Default 256.
    dim : int
        Vector dimension. Must be a power of two, >= 128 (LASER constraint).
    R : int
        Graph degree bound for the underlying Vamana / QG. Default 64.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    tuple
        `(prefix, vectors, labels)` where `prefix` is the
        `dsqg_<seg_basename>_R<R>_MD<dim>` filename prefix written into
        `target_dir`, `vectors` is the `(n, dim)` float32 array used to
        build the index, and `labels` is a `(n,)` uint64 array of
        synthetic external labels.

    Raises
    ------
    RuntimeError
        If `alayalite.laser` or `alayalite.vamana` cannot be imported on
        the current build (i.e., the wheel was built without LASER).
    """
    if dim < 128 or dim & (dim - 1):
        raise ValueError(f"dim must be a power of two and >= 128 (got {dim})")
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        # pylint: disable=import-outside-toplevel
        # Intentional lazy import: alayalite.laser / vamana are not loadable on
        # builds without ALAYA_ENABLE_LASER=ON. Tests gate the call on
        # _laser_support.DISK_LASER_SUPPORTED so this branch only fires when
        # the modules are present; the except below stays for defence in
        # depth in case the runtime probe and the actual import disagree.
        from alayalite import laser as laser_module
    except ImportError as exc:
        raise RuntimeError(
            "build_small_laser_artifacts requires alayalite.laser, "
            "which are unavailable on this build. Skip via _laser_support.DISK_LASER_SUPPORTED "
            "before calling."
        ) from exc

    main_dim = dim
    prefix = f"dsqg_{seg_basename}_R{R}_MD{main_dim}"
    qg_prefix = f"dsqg_{seg_basename}"

    vectors = _generate_vectors(n, dim, seed)
    laser_module.Index.fit(
        vectors,
        output_dir=target_dir,
        name=qg_prefix,
        build_params=laser_module.BuildParams(
            metric="l2",
            main_dim=main_dim,
            R=R,
            L=max(R + 8, 100),
            alpha=1.2,
            ef_indexing=200,
            ep_num=max(1, min(16, n)),
            disable_medoid=True,
        ),
        num_threads=1,
        seed=seed,
        skip_existing=False,
        auto_load=False,
        dram_budget_gb=1.0,
    )

    required_paths = [
        target_dir / f"{prefix}.index",
        target_dir / f"{prefix}.index_rotator",
        target_dir / f"{prefix}.index_cache_ids",
        target_dir / f"{prefix}.index_cache_nodes",
    ]
    for p in required_paths:
        if not p.is_file() or p.stat().st_size <= 0:
            raise RuntimeError(f"LASER fixture artifact missing or empty after build: {p}")

    # Verify the index file's first qword equals `n` (matching the
    # C++ importer's `read_index_count` contract).
    with required_paths[0].open("rb") as f:
        head = f.read(8)
    declared_n = struct.unpack("<Q", head)[0]
    if declared_n != n:
        raise RuntimeError(f"LASER index declared count={declared_n} differs from requested n={n}")

    # Synthetic labels: spaced apart so tests that assert "external labels are
    # not internal PIDs (0..n-1)" see a clear separation.
    labels = (1_000_000 + np.arange(n, dtype=np.uint64)).astype(np.uint64)
    return prefix, vectors, labels
