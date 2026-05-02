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

"""Manual smoke benchmark for alayalite.DiskCollection(index_type="disk_laser").

Runs end-to-end against a precomputed LASER artifacts directory provided
via ``--src-dir`` and prints recall@k, p50, p95 latency, QPS,
import_seconds, and segment_bytes — mirroring `disk_vamana_smoke.py` so
grep-based tooling works for both engines.

Recall is measured against:
- a `disk_flat` collection built in-process when ``--vectors`` points at
  the source `.fbin` file (highest fidelity), or
- a precomputed top-k labels file passed via ``--ground-truth``, or
- ``N/A`` when neither is available (a warning is emitted).

This script is NOT wired into CI and does NOT have a performance gate.
On unsupported builds it gracefully prints
"disk_laser not available on this build" and exits ``0`` so the same
invocation is safe across the wheel matrix.

NOTE: the wheel must be rebuilt for the target CPU architecture; the
Python extension is built with ``-march=native`` and AVX-512 wheels do
not load on CPUs without AVX-512 (this is a pre-existing wheel-
distribution concern, not introduced by this benchmark).
"""

from __future__ import annotations

import argparse
import shutil
import struct
import tempfile
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from alayalite import DiskCollection, MetricType


def _load_fbin(path: Path) -> np.ndarray:
    """Read DiskANN-format `.fbin`: little-endian uint32 num, uint32 dim, then num*dim float32."""
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError(f"truncated .fbin header at {path}")
        num, dim = struct.unpack("<II", header)
        body = np.fromfile(f, dtype=np.float32, count=num * dim)
        if body.shape[0] != num * dim:
            raise RuntimeError(f"truncated .fbin body at {path}: expected {num * dim} floats, got {body.shape[0]}")
    return np.ascontiguousarray(body.reshape(num, dim))


def _load_ground_truth(path: Path) -> dict[int, set[int]]:
    """Read DiskANN-style `.bin` with uint32 payload (top-k labels per query)."""
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError(f"truncated ground-truth header at {path}")
        nq, k = struct.unpack("<II", header)
        body = np.fromfile(f, dtype=np.uint32, count=nq * k)
    body = body.reshape(nq, k)
    return {i: {int(v) for v in body[i]} for i in range(nq)}


def _build_flat_for_recall(
    work_dir: Path,
    vectors: np.ndarray,
    labels: np.ndarray,
) -> DiskCollection:
    flat = DiskCollection(
        path=str(work_dir),
        dim=vectors.shape[1],
        metric=MetricType.L2,
        index_type="disk_flat",
    )
    flat.add(vectors, labels)
    flat.flush()
    return flat


def _build_laser(
    work_dir: Path,
    src_dir: Path,
    labels: np.ndarray,
    dim: int,
) -> tuple[DiskCollection, float]:
    laser = DiskCollection(
        path=str(work_dir),
        dim=dim,
        metric=MetricType.L2,
        index_type="disk_laser",
    )
    start = time.perf_counter()
    laser.import_laser_segment(str(src_dir), labels)
    return laser, time.perf_counter() - start


def _segment_size_bytes(collection_path: Path) -> int:
    segments = collection_path / "segments"
    return sum(p.stat().st_size for p in segments.rglob("*") if p.is_file())


def _run_queries(
    laser: DiskCollection,
    queries: np.ndarray,
    k: int,
    ef: int,
    beam_width: int,
    flat: Optional[DiskCollection],
    ground_truth: Optional[dict[int, set[int]]],
) -> tuple[Optional[float], np.ndarray]:
    recalls: Optional[list[float]]
    if flat is not None:
        recalls = []
        latencies_ms = []
        for query in queries:
            exact = {label for label, _ in flat.search(query, k=k)}
            start = time.perf_counter()
            approx = {label for label, _ in laser.search(query, k=k, ef=ef, beam_width=beam_width)}
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            recalls.append(len(exact & approx) / float(k))
        return float(np.mean(recalls)), np.asarray(latencies_ms, dtype=np.float64)
    if ground_truth is not None:
        recalls = []
        latencies_ms = []
        for i, query in enumerate(queries):
            start = time.perf_counter()
            approx = {label for label, _ in laser.search(query, k=k, ef=ef, beam_width=beam_width)}
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            exact = ground_truth.get(i, set())
            recalls.append(len(exact & approx) / float(k))
        return float(np.mean(recalls)), np.asarray(latencies_ms, dtype=np.float64)
    # No recall reference. Report N/A.
    latencies_ms = []
    for query in queries:
        start = time.perf_counter()
        _ = laser.search(query, k=k, ef=ef, beam_width=beam_width)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return None, np.asarray(latencies_ms, dtype=np.float64)


def _generate_queries(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True, help="Precomputed LASER artifacts directory.")
    parser.add_argument("--n", type=int, required=True, help="Number of vectors in the LASER index.")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension.")
    parser.add_argument(
        "--vectors", type=Path, default=None, help="Optional .fbin source vectors for disk_flat ground truth."
    )
    parser.add_argument("--ground-truth", type=Path, default=None, help="Optional precomputed top-k labels file.")
    parser.add_argument("--queries-path", type=Path, default=None, help="Optional .fbin queries file.")
    parser.add_argument(
        "--queries", type=int, default=100, help="Number of synthetic queries (when --queries-path is omitted)."
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef", type=int, default=128)
    parser.add_argument("--beam-width", dest="beam_width", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory for disk_laser/disk_flat collections.")
    parser.add_argument("--keep", action="store_true", help="Keep generated work directory.")
    return parser.parse_args()


def main() -> int:
    try:
        args = _parse_args()
        if args.dim < 128 or args.dim & (args.dim - 1):
            raise SystemExit("--dim must be a power of two and >= 128 for the LASER engine.")

        tmp_ctx = None
        if args.work_dir is None:
            if args.keep:
                work_dir = Path(tempfile.mkdtemp(prefix="alayalite_disk_laser_smoke_"))
            else:
                tmp_ctx = tempfile.TemporaryDirectory(prefix="alayalite_disk_laser_smoke_")
                work_dir = Path(tmp_ctx.name)
        else:
            work_dir = args.work_dir
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True)

        try:
            labels = (1_000_000 + np.arange(args.n, dtype=np.uint64)).astype(np.uint64)

            flat = None
            if args.vectors is not None:
                vectors = _load_fbin(args.vectors)
                if vectors.shape[0] != args.n:
                    raise SystemExit(f"--vectors row count ({vectors.shape[0]}) does not match --n ({args.n})")
                if vectors.shape[1] != args.dim:
                    raise SystemExit(f"--vectors dim ({vectors.shape[1]}) does not match --dim ({args.dim})")
                flat = _build_flat_for_recall(work_dir / "disk_flat", vectors, labels)

            ground_truth = None
            if args.ground_truth is not None:
                ground_truth = _load_ground_truth(args.ground_truth)

            laser, import_seconds = _build_laser(
                work_dir / "disk_laser",
                args.src_dir,
                labels,
                args.dim,
            )

            if args.queries_path is not None:
                queries = _load_fbin(args.queries_path)
                if queries.shape[1] != args.dim:
                    raise SystemExit(f"--queries-path dim ({queries.shape[1]}) does not match --dim ({args.dim})")
            else:
                queries = _generate_queries(args.queries, args.dim, args.seed)

            recall, latencies_ms = _run_queries(
                laser,
                queries,
                args.k,
                args.ef,
                args.beam_width,
                flat,
                ground_truth,
            )

            total_seconds = float(latencies_ms.sum() / 1000.0)
            qps = len(latencies_ms) / total_seconds if total_seconds > 0 else float("inf")
            p50 = float(np.percentile(latencies_ms, 50))
            p95 = float(np.percentile(latencies_ms, 95))
            segment_bytes = _segment_size_bytes(work_dir / "disk_laser")

            if recall is None:
                warnings.warn(
                    "neither --vectors nor --ground-truth were provided; recall will be reported as N/A",
                    RuntimeWarning,
                    stacklevel=1,
                )
                print(f"recall@{args.k}: N/A")
            else:
                print(f"recall@{args.k}: {recall:.4f}")
            print(f"p50_ms: {p50:.3f}")
            print(f"p95_ms: {p95:.3f}")
            print(f"qps: {qps:.2f}")
            print(f"import_seconds: {import_seconds:.3f}")
            print(f"segment_bytes: {segment_bytes}")
            print(f"work_dir: {work_dir}")
            return 0
        finally:
            if tmp_ctx is not None and not args.keep:
                tmp_ctx.cleanup()
    except (ValueError, RuntimeError) as exc:
        msg = str(exc)
        if "disk_laser" in msg and "not implemented in v1" in msg:
            print("disk_laser not available on this build")
            return 0
        raise


if __name__ == "__main__":
    raise SystemExit(main())
