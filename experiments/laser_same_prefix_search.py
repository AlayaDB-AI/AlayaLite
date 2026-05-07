"""Same-prefix dual-load probe.

Loads two LASER Index instances from the SAME prefix (the manual one) and
measures QPS for each, in the same order/protocol the bench uses. If even
two indices pointing at the same physical files diverge by ~10%, the
issue is per-instance (io_context resource contention, ThreadData
allocation order, etc.). If they agree within ~1%, the unified-index
slowdown is rooted in the underlying file's physical layout (ext4 extent
allocation / SSD LBA placement) — not anything we wrote.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from alayalite import laser
from alayalite.laser.io import read_fbin


def _measure(idx, queries, *, k, ef, threads, beam, warmup, runs):
    idx.set_params(ef_search=int(ef), num_threads=int(threads), beam_width=int(beam))
    for _ in range(warmup):
        idx.batch_search(queries, int(k))
    t0 = time.perf_counter()
    for _ in range(runs):
        idx.batch_search(queries, int(k))
    return float(queries.shape[0] * runs / (time.perf_counter() - t0))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True, type=Path, help="LASER prefix to load (without _R*_MD*.index suffix)")
    p.add_argument("--query-fbin", required=True, type=Path)
    p.add_argument("--efs", default="80,100,150,200,300,500")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--dram", type=float, default=2.0)
    args = p.parse_args()

    queries = np.ascontiguousarray(read_fbin(str(args.query_fbin), use_mmap=False), dtype=np.float32)
    efs = [int(x) for x in args.efs.split(",") if x]

    print(f"Loading idx_a from {args.prefix}")
    idx_a = laser.Index.from_prefix(str(args.prefix), dram_budget_gb=args.dram)
    print(f"Loading idx_b from {args.prefix} (same physical files)")
    idx_b = laser.Index.from_prefix(str(args.prefix), dram_budget_gb=args.dram)

    print(f"\n  {'EF':>4}  {'idx_a':>9}  {'idx_b':>9}  {'Δ %':>7}")
    for ef in efs:
        a = _measure(
            idx_a, queries, k=args.k, ef=ef, threads=args.threads, beam=args.beam, warmup=args.warmup, runs=args.runs
        )
        b = _measure(
            idx_b, queries, k=args.k, ef=ef, threads=args.threads, beam=args.beam, warmup=args.warmup, runs=args.runs
        )
        print(f"  {ef:>4}  {a:>9.2f}  {b:>9.2f}  {(b - a) / a * 100:>+6.2f}%")


if __name__ == "__main__":
    main()
