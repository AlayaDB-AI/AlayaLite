"""Swap-order search probe for the LASER unified-fit bench.

Loads the manual+unified LASER indices already built by
`laser_unified_fit_bench.py` and re-measures QPS twice:

    pass A: manual first, unified second (matches bench order)
    pass B: unified first, manual second (reversed)

If pass B's QPS delta flips sign vs pass A, the ~10% gap is first-mover
advantage (CPU L*/SSD prefetcher/io_context state) — not a bench-side
implementation bug. If pass B keeps the same sign, the disk artifacts
have a hidden divergence and we need to widen the diff.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from alayalite import laser
from alayalite.laser._io import read_fbin


def _measure(idx, queries: np.ndarray, *, k: int, ef: int, threads: int, beam: int, warmup: int, runs: int) -> float:
    idx.set_params(ef_search=int(ef), num_threads=int(threads), beam_width=int(beam))
    for _ in range(warmup):
        idx.batch_search(queries, int(k))
    t0 = time.perf_counter()
    for _ in range(runs):
        idx.batch_search(queries, int(k))
    elapsed = time.perf_counter() - t0
    return float(queries.shape[0] * runs / elapsed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repeat-dir", required=True, type=Path, help="path to .../repeat_0 (containing manual/ and unified/)"
    )
    p.add_argument("--query-fbin", required=True, type=Path)
    p.add_argument("--manual-name", default="gist1m_manual")
    p.add_argument("--unified-name", default="gist1m_unified")
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

    manual_prefix = str(args.repeat_dir / "manual" / args.manual_name)
    unified_prefix = str(args.repeat_dir / "unified" / args.unified_name)

    print("== Pass A: manual first, unified second ==")
    manual_a = laser.Index.from_prefix(manual_prefix, dram_budget_gb=args.dram)
    unified_a = laser.Index.from_prefix(unified_prefix, dram_budget_gb=args.dram)
    rows_a = []
    for ef in efs:
        m = _measure(
            manual_a, queries, k=args.k, ef=ef, threads=args.threads, beam=args.beam, warmup=args.warmup, runs=args.runs
        )
        u = _measure(
            unified_a,
            queries,
            k=args.k,
            ef=ef,
            threads=args.threads,
            beam=args.beam,
            warmup=args.warmup,
            runs=args.runs,
        )
        delta = (u - m) / m * 100.0
        rows_a.append((ef, m, u, delta))
        print(f"  EF={ef:3d}  manual={m:8.2f}  unified={u:8.2f}  Δ={delta:+6.2f}%")
    del manual_a, unified_a

    print("\n== Pass B: unified first, manual second ==")
    unified_b = laser.Index.from_prefix(unified_prefix, dram_budget_gb=args.dram)
    manual_b = laser.Index.from_prefix(manual_prefix, dram_budget_gb=args.dram)
    rows_b = []
    for ef in efs:
        u = _measure(
            unified_b,
            queries,
            k=args.k,
            ef=ef,
            threads=args.threads,
            beam=args.beam,
            warmup=args.warmup,
            runs=args.runs,
        )
        m = _measure(
            manual_b, queries, k=args.k, ef=ef, threads=args.threads, beam=args.beam, warmup=args.warmup, runs=args.runs
        )
        delta = (u - m) / m * 100.0
        rows_b.append((ef, m, u, delta))
        print(f"  EF={ef:3d}  manual={m:8.2f}  unified={u:8.2f}  Δ={delta:+6.2f}%")

    print("\n== Δ comparison ==")
    print(f"  {'EF':>4}  {'Δ pass A':>10}  {'Δ pass B':>10}  {'sign flip?':>11}")
    for (ef, _, _, da), (_, _, _, db) in zip(rows_a, rows_b):
        flip = "YES" if (da * db < 0) else "no"
        print(f"  {ef:>4}  {da:>+9.2f}%  {db:>+9.2f}%  {flip:>11}")


if __name__ == "__main__":
    main()
