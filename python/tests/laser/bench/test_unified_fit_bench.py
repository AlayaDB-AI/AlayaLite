# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Smoke test for shared-vamana mode of `laser_unified_fit_bench.py`.

Asserts that with the bench in shared-vamana mode (the default after this
change), the manual and unified pipelines produce byte-equal LASER
artifacts, regardless of vamana's multi-thread non-determinism.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _laser_support import DISK_LASER_SUPPORTED

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DISK_LASER_SUPPORTED,
        reason="disk_laser is not supported on this build/platform",
    ),
]


def _write_fbin(path: Path, vectors: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array([vectors.shape[0], vectors.shape[1]], dtype=np.int32).tofile(f)
        vectors.astype(np.float32, copy=False).tofile(f)


def _write_ibin(path: Path, ids: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array([ids.shape[0], ids.shape[1]], dtype=np.int32).tofile(f)
        ids.astype(np.int32, copy=False).tofile(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _hashes_for(prefix_dir: Path, name: str, graph_r: int, main_dim: int) -> dict[str, str]:
    suffixes = [
        f"R{graph_r}_MD{main_dim}.index",
        f"R{graph_r}_MD{main_dim}.index_rotator",
        f"R{graph_r}_MD{main_dim}.index_cache_ids",
        f"R{graph_r}_MD{main_dim}.index_cache_nodes",
    ]
    return {s: _sha256(prefix_dir / f"{name}_{s}") for s in suffixes}


def test_bench_shared_vamana_produces_byte_equal_artifacts(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    n_base, n_query, dim = 512, 16, 128
    base = rng.normal(size=(n_base, dim)).astype(np.float32)
    queries = rng.normal(size=(n_query, dim)).astype(np.float32)
    # Top-1 ground truth: just int32 indices, content irrelevant for this smoke.
    gt = np.zeros((n_query, 10), dtype=np.int32)

    dataset_root = tmp_path / "data"
    dataset_root.mkdir()
    _write_fbin(dataset_root / "gist_base.fbin", base)
    _write_fbin(dataset_root / "gist_query.fbin", queries)
    _write_ibin(dataset_root / "gist_gt.ibin", gt)

    output_dir = tmp_path / "out"
    run_id = "smoke"

    # main_dim == dim disables PCA (cleaner smoke); disable_medoid avoids
    # the medoid step which needs sklearn/faiss in some envs.
    # R=64 is required: with dim=128, node_len_=(32*128+32*128+128*64+64*128)/8=3072
    # which satisfies node_len_ >= kSectorLen/2 so node_per_page_==1. R=32 gives
    # node_len_=2048 < kSectorLen=4096 → node_per_page_=2 → ValueError at construction.
    graph_r = 64
    completed = subprocess.run(
        [
            sys.executable,
            "python/benchmarks/laser_unified_fit_bench.py",
            "--dataset-root",
            str(dataset_root),
            "--datasets",
            "gist1m",
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id,
            "--main-dim",
            str(dim),
            "--R",
            str(graph_r),
            "--L",
            "64",
            "--alpha",
            "1.2",
            "--threads",
            "4",
            "--ef-indexing",
            "100",
            "--efs",
            "50",
            "--k",
            "10",
            "--warmup",
            "1",
            "--runs",
            "2",
            "--build-repeats",
            "1",
            "--ep-num",
            "8",
            "--seed",
            "42",
            "--pca-seed",
            "42",
            "--medoid-seed",
            "42",
            "--vamana-seed",
            "42",
            "--rotator-seed",
            "42",
            "--build-dram-budget-gb",
            "1.0",
            "--search-dram-budget-gb",
            "1.0",
            "--disable-medoid",
            "--keep-artifacts",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"bench exited {completed.returncode}\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
    )

    repeat_dir = output_dir / run_id / "gist1m" / "repeat_0"
    manual_dir = repeat_dir / "manual"
    unified_dir = repeat_dir / "unified"
    assert manual_dir.is_dir() and unified_dir.is_dir()

    manual_hashes = _hashes_for(manual_dir, "gist1m_manual", graph_r=graph_r, main_dim=dim)
    unified_hashes = _hashes_for(unified_dir, "gist1m_unified", graph_r=graph_r, main_dim=dim)

    # Strip prefix so name-mismatch doesn't trip equality
    manual_by_suffix = dict(manual_hashes.items())
    unified_by_suffix = dict(unified_hashes.items())
    assert manual_by_suffix == unified_by_suffix, (
        "shared-vamana mode should make manual/unified LASER artifacts byte-equal; "
        f"got manual={manual_by_suffix}, unified={unified_by_suffix}"
    )

    # JSON report should expose vamana wall time as a separate field.
    report = json.loads((output_dir / run_id / "report.json").read_text())
    ds_report = report["reports"][0]
    assert "vamana_shared_s" in ds_report, (
        f"expected vamana_shared_s in dataset report, got keys: {sorted(ds_report.keys())}"
    )
    assert ds_report["vamana_shared_s"] >= 0.0
