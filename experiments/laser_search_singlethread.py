"""Single-threaded search probe — reuses an existing LASER prefix.

Loads the LASER index built by `laser_unified_fit_bench.py`, runs
`batch_search` with `num_threads=1` across a sweep of EF values, and prints
recall + QPS in the same shape as the bench's markdown table.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from alayalite import laser
from alayalite.laser._io import read_fbin, read_ibin


def _recall_at_k(predictions: np.ndarray, gt: np.ndarray, k: int) -> float:
    correct = 0
    nq = predictions.shape[0]
    for i in range(nq):
        correct += len(set(predictions[i, :k].tolist()) & set(gt[i, :k].tolist()))
    return correct / (nq * k)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True, type=Path)
    p.add_argument("--query-fbin", required=True, type=Path)
    p.add_argument("--gt-ibin", required=True, type=Path)
    p.add_argument("--efs", default="80,90,100,110,130,150,200,250,300,400,500")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--dram", type=float, default=2.0)
    args = p.parse_args()

    queries = np.ascontiguousarray(read_fbin(str(args.query_fbin), use_mmap=False), dtype=np.float32)
    gt = np.asarray(read_ibin(str(args.gt_ibin)), dtype=np.int32)
    efs = [int(x) for x in args.efs.split(",") if x]

    idx = laser.Index.from_prefix(str(args.prefix), dram_budget_gb=args.dram)

    print(f"\n  {'EF':>4}  {'Recall':>8}  {'QPS':>10}")
    for ef in efs:
        idx.set_params(ef_search=ef, num_threads=1, beam_width=args.beam)
        for _ in range(args.warmup):
            idx.batch_search(queries, args.k)

        preds = idx.batch_search(queries, args.k)
        recall = _recall_at_k(preds, gt, args.k)

        t0 = time.perf_counter()
        for _ in range(args.runs):
            idx.batch_search(queries, args.k)
        elapsed = time.perf_counter() - t0
        qps = queries.shape[0] * args.runs / elapsed

        print(f"  {ef:>4}  {recall:>8.4f}  {qps:>10.2f}")


if __name__ == "__main__":
    main()
