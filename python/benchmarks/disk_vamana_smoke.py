#!/usr/bin/env python3
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

"""Manual smoke benchmark for alayalite.DiskCollection(index_type="disk_vamana")."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from alayalite import DiskCollection, MetricType


def _generate_data(n: int, dim: int, query_count: int, seed: int):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((query_count, dim)).astype(np.float32)
    labels = (1_000_000 + np.arange(n, dtype=np.uint64)).astype(np.uint64)
    return vectors, queries, labels


def _build_flat(path: Path, vectors: np.ndarray, labels: np.ndarray) -> DiskCollection:
    col = DiskCollection(
        path=str(path),
        dim=vectors.shape[1],
        metric=MetricType.L2,
        index_type="disk_flat",
    )
    col.add(vectors, labels)
    col.flush()
    return col


def _build_vamana(path: Path, vectors: np.ndarray, labels: np.ndarray, args) -> tuple[DiskCollection, float]:
    col = DiskCollection(
        path=str(path),
        dim=vectors.shape[1],
        metric=MetricType.L2,
        index_type="disk_vamana",
        vamana_R=args.vamana_R,
        vamana_L=args.vamana_L,
        vamana_alpha=args.vamana_alpha,
        vamana_seed=args.vamana_seed,
    )
    start = time.perf_counter()
    col.add(vectors, labels)
    col.flush()
    return col, time.perf_counter() - start


def _segment_size_bytes(collection_path: Path) -> int:
    segments = collection_path / "segments"
    return sum(path.stat().st_size for path in segments.rglob("*") if path.is_file())


def _run_queries(flat: DiskCollection, vamana: DiskCollection, queries: np.ndarray, k: int, ef: int):
    recalls = []
    latencies_ms = []
    for query in queries:
        exact = {label for label, _ in flat.search(query, k=k)}
        start = time.perf_counter()
        approx = {label for label, _ in vamana.search(query, k=k, ef=ef)}
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        recalls.append(len(exact & approx) / float(k))
    return float(np.mean(recalls)), np.asarray(latencies_ms, dtype=np.float64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory for benchmark collections.")
    parser.add_argument("--keep", action="store_true", help="Keep the generated work directory.")
    parser.add_argument("--n", type=int, default=10_000, help="Number of vectors.")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension.")
    parser.add_argument("--queries", type=int, default=100, help="Number of deterministic queries.")
    parser.add_argument("--k", type=int, default=10, help="Recall/search depth.")
    parser.add_argument("--ef", type=int, default=128, help="Vamana search ef.")
    parser.add_argument("--seed", type=int, default=12345, help="Dataset seed.")
    parser.add_argument("--vamana-R", dest="vamana_R", type=int, default=64)
    parser.add_argument("--vamana-L", dest="vamana_L", type=int, default=100)
    parser.add_argument("--vamana-alpha", dest="vamana_alpha", type=float, default=1.2)
    parser.add_argument("--vamana-seed", dest="vamana_seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tmp_ctx = None
    if args.work_dir is None:
        if args.keep:
            work_dir = Path(tempfile.mkdtemp(prefix="alayalite_disk_vamana_smoke_"))
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="alayalite_disk_vamana_smoke_")
            work_dir = Path(tmp_ctx.name)
    else:
        work_dir = args.work_dir
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)

    try:
        vectors, queries, labels = _generate_data(args.n, args.dim, args.queries, args.seed)
        flat = _build_flat(work_dir / "disk_flat", vectors, labels)
        vamana, build_seconds = _build_vamana(work_dir / "disk_vamana", vectors, labels, args)
        recall, latencies_ms = _run_queries(flat, vamana, queries, args.k, args.ef)
        total_seconds = float(latencies_ms.sum() / 1000.0)
        qps = len(latencies_ms) / total_seconds if total_seconds > 0 else float("inf")
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        segment_bytes = _segment_size_bytes(work_dir / "disk_vamana")

        print(f"recall@{args.k}: {recall:.4f}")
        print(f"p50_ms: {p50:.3f}")
        print(f"p95_ms: {p95:.3f}")
        print(f"qps: {qps:.2f}")
        print(f"vamana_build_seconds: {build_seconds:.3f}")
        print(f"vamana_segment_bytes: {segment_bytes}")
        print(f"work_dir: {work_dir}")
    finally:
        if tmp_ctx is not None and not args.keep:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
