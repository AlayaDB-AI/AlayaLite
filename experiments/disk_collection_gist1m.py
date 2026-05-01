# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""GIST 1M experiment for alayalite.DiskCollection vs numpy brute-force.

Loads `gist_base.fbin` (1M x 960 float32) and `gist_query.fbin` (1000 queries),
ingests into a `DiskCollection` (forced to multiple segments because the
default 512 MiB pending cap cannot hold 3.84 GB of vectors at once), then
runs all 1000 queries to measure:

- recall@10 vs ground truth (`gist_gt.ibin`)
- QPS, p50/p95/p99 latency
- build/flush wall time
- peak RSS

A numpy `argpartition`-based reference is run on the same queries to
provide an apples-to-apples ratio. Results are written to the directory
passed via `--out` (default `/md1/huangliang/alaya-dev/perf`).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from alayalite import DiskCollection, MetricType

# ---- Format helpers -------------------------------------------------------


def load_fbin(path: str) -> np.ndarray:
    """Load an .fbin file as a (N, D) float32 array. Memory-mapped for the
    full base set so we don't double the working set."""
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.int32)
    n, d = int(header[0]), int(header[1])
    arr = np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, d))
    return arr


def load_ibin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.int32)
    n, k = int(header[0]), int(header[1])
    arr = np.memmap(path, dtype=np.int32, mode="r", offset=8, shape=(n, k))
    return arr


# ---- Recall ---------------------------------------------------------------


def recall_at_k(predicted_ids: list[int], ground_truth: np.ndarray, k: int) -> float:
    gt_topk = {int(x) for x in ground_truth[:k]}
    pred_topk = {int(x) for x in predicted_ids[:k]}
    return len(gt_topk & pred_topk) / k


# ---- Timing helpers -------------------------------------------------------


def percentiles(values: list[float], qs=(50, 95, 99)) -> dict[int, float]:
    s = sorted(values)
    out: dict[int, float] = {}
    for q in qs:
        idx = max(0, min(len(s) - 1, int(len(s) * q / 100)))
        out[q] = s[idx]
    return out


# ---- Hardware / software provenance ---------------------------------------


def hw_provenance() -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
                if line.startswith("flags"):
                    flags = line.split(":", 1)[1].split()
                    info["simd_flags"] = sorted(f for f in flags if f.startswith(("sse4", "avx")))
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["mem_total_kb"] = int(line.split()[1])
                    break
    except OSError:
        pass
    return info


# ---- Main -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/md1/huangliang/alaya-dev/data/gist1m")
    parser.add_argument("--out", default="/md1/huangliang/alaya-dev/perf")
    parser.add_argument(
        "--coll-dir",
        default=None,
        help="Where to write the DiskCollection. Default = a tempdir under perf/, kept after run for inspection.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50_000, help="Rows per add_batch + flush; bounded by 512 MiB pending cap"
    )
    parser.add_argument("--n-queries", type=int, default=1000, help="Limit timed queries (default = all 1000)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--metric",
        choices=["L2"],
        default="L2",
        help="GIST 1M ground truth is L2 — that's the only metric we evaluate recall against",
    )
    parser.add_argument("--skip-numpy", action="store_true", help="Skip numpy reference (saves ~30-60s)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    print(f"[{datetime.now().isoformat(timespec='seconds')}] loading dataset...")
    base = load_fbin(os.path.join(args.data_dir, "gist_base.fbin"))
    queries = load_fbin(os.path.join(args.data_dir, "gist_query.fbin"))
    gt = load_ibin(os.path.join(args.data_dir, "gist_gt.ibin"))
    n, dim = base.shape
    n_queries = min(args.n_queries, queries.shape[0])
    assert dim == 960
    assert gt.shape[0] >= n_queries
    print(f"  base: {base.shape} {base.dtype}")
    print(f"  queries: {queries.shape}")
    print(f"  gt: {gt.shape}")

    # ---- Build phase ------------------------------------------------------

    coll_dir = args.coll_dir
    if coll_dir is None:
        coll_dir = str(out_dir / f"disk_coll_{ts}")
    if os.path.exists(coll_dir):
        import shutil

        shutil.rmtree(coll_dir)

    print(f"[{datetime.now().isoformat(timespec='seconds')}] building DiskCollection at {coll_dir}")
    print(f"  batches of {args.batch_size}, expecting {(n + args.batch_size - 1) // args.batch_size} segments")

    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    metric = MetricType.L2  # GIST gt is L2
    col = DiskCollection(path=coll_dir, dim=dim, metric=metric, index_type="disk_flat")

    build_t_start = time.perf_counter()
    for offset in range(0, n, args.batch_size):
        end = min(offset + args.batch_size, n)
        batch_v = np.ascontiguousarray(base[offset:end])
        batch_ids = np.arange(offset, end, dtype=np.uint64)
        col.add(batch_v, batch_ids)
        col.flush()
        if (offset // args.batch_size) % 5 == 0:
            print(f"  flushed up to row {end} ({end / n * 100:.0f}%)")
    build_s = time.perf_counter() - build_t_start
    print(f"  build/flush total: {build_s:.1f}s, size={col.size()}")
    assert col.size() == n
    rss_after_build_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ---- Search phase: DiskCollection ------------------------------------

    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] DiskCollection search loop "
        f"({n_queries} queries, k={args.k}, +{args.warmup} warmup)"
    )
    # Warmup
    for i in range(args.warmup):
        col.search(np.ascontiguousarray(queries[i]), k=args.k)

    coll_recalls: list[float] = []
    coll_lat_us: list[float] = []
    t_start = time.perf_counter()
    for i in range(n_queries):
        q = np.ascontiguousarray(queries[i])
        t = time.perf_counter()
        hits = col.search(q, k=args.k)
        coll_lat_us.append((time.perf_counter() - t) * 1e6)
        pred_ids = [h[0] for h in hits]
        coll_recalls.append(recall_at_k(pred_ids, gt[i], args.k))
    coll_total_s = time.perf_counter() - t_start
    coll_qps = n_queries / coll_total_s
    coll_p = percentiles(coll_lat_us)
    coll_recall = statistics.mean(coll_recalls)

    # ---- Reference phase: numpy bf ---------------------------------------

    np_qps = None
    np_p = None
    np_recalls: list[float] | None = None
    np_lat_us: list[float] | None = None
    if not args.skip_numpy:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] numpy bf reference ({n_queries} queries, k={args.k})")
        # Warmup
        for i in range(args.warmup):
            d = base - queries[i]
            np.argpartition((d * d).sum(axis=1), args.k)[: args.k]

        np_recalls = []
        np_lat_us = []
        t_start = time.perf_counter()
        for i in range(n_queries):
            t = time.perf_counter()
            d = base - queries[i]
            ssd = (d * d).sum(axis=1)
            top_idx = np.argpartition(ssd, args.k)[: args.k]
            order = top_idx[np.argsort(ssd[top_idx])]
            np_lat_us.append((time.perf_counter() - t) * 1e6)
            np_recalls.append(recall_at_k(order.tolist(), gt[i], args.k))
        np_total_s = time.perf_counter() - t_start
        np_qps = n_queries / np_total_s
        np_p = percentiles(np_lat_us)

    rss_peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ---- Report -----------------------------------------------------------

    md_path = out_dir / f"gist1m_disk_collection_{ts}.md"
    json_path = out_dir / f"gist1m_disk_collection_{ts}.json"
    hw = hw_provenance()
    n_segments = (n + args.batch_size - 1) // args.batch_size

    md_lines = [
        "# DiskCollection on GIST 1M",
        "",
        f"- timestamp: {ts}",
        f"- dataset: gist1m (n={n}, dim={dim}, metric={args.metric})",
        f"- queries: {n_queries} timed (+{args.warmup} warmup), k={args.k}",
        f"- batches: {args.batch_size} rows per flush → {n_segments} segments",
        f"- collection path: `{coll_dir}` (kept after run)",
        f"- cpu: {hw.get('cpu_model', 'unknown')}",
        f"- mem_total_kb: {hw.get('mem_total_kb')}",
        f"- simd_flags: {' '.join(hw.get('simd_flags', []))}",
        f"- python: {hw.get('python')}",
        "- compiler flags: -Ofast -DNDEBUG (release)",
        "",
        "## Build",
        "",
        f"- build/flush total: {build_s:.2f}s ({n / build_s:.0f} rows/s)",
        f"- segments produced: {n_segments}",
        f"- RSS before build: {rss_before_kb / 1024:.0f} MiB",
        f"- RSS after build: {rss_after_build_kb / 1024:.0f} MiB",
        f"- RSS peak (process): {rss_peak_kb / 1024:.0f} MiB",
        "",
        "## Search",
        "",
        "| metric | DiskCollection | numpy bf | DC/np |",
        "|---|---:|---:|---:|",
        f"| recall@{args.k} | {coll_recall:.4f} | {statistics.mean(np_recalls) if np_recalls else 'n/a'} | n/a |",
        f"| QPS | {coll_qps:.1f} | "
        f"{f'{np_qps:.1f}' if np_qps else 'n/a'} | "
        f"{f'{coll_qps / np_qps:.2f}x' if np_qps else 'n/a'} |",
        f"| p50 (us) | {coll_p[50]:.0f} | "
        f"{f'{np_p[50]:.0f}' if np_p else 'n/a'} | "
        f"{f'{coll_p[50] / np_p[50]:.2f}x' if np_p else 'n/a'} |",
        f"| p95 (us) | {coll_p[95]:.0f} | "
        f"{f'{np_p[95]:.0f}' if np_p else 'n/a'} | "
        f"{f'{coll_p[95] / np_p[95]:.2f}x' if np_p else 'n/a'} |",
        f"| p99 (us) | {coll_p[99]:.0f} | "
        f"{f'{np_p[99]:.0f}' if np_p else 'n/a'} | "
        f"{f'{coll_p[99] / np_p[99]:.2f}x' if np_p else 'n/a'} |",
        f"| min (us) | {min(coll_lat_us):.0f} | {f'{min(np_lat_us):.0f}' if np_lat_us else 'n/a'} | n/a |",
        f"| mean (us) | {statistics.mean(coll_lat_us):.0f} | "
        f"{f'{statistics.mean(np_lat_us):.0f}' if np_lat_us else 'n/a'} | n/a |",
        "",
        "## Notes",
        "",
        "- Both phases run single-threaded.",
        "- Recall is computed against the dataset's published top-100 ground truth (k-truncated to args.k).",
        "- DiskCollection runs through `mmap`'d on-disk segments; cold-cache "
        "effects are absorbed by the warmup queries.",
        f"- {n_segments} segments at {args.batch_size} rows each forces a "
        "global merge across all segments per query — exercises the multi-segment "
        "search path explicitly.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n")

    json_payload = {
        "timestamp": ts,
        "dataset": {"name": "gist1m", "n": n, "dim": dim, "metric": args.metric},
        "queries": {"timed": n_queries, "warmup": args.warmup, "k": args.k},
        "config": {
            "batch_size": args.batch_size,
            "n_segments": n_segments,
            "coll_path": coll_dir,
        },
        "build": {
            "wall_seconds": build_s,
            "rows_per_second": n / build_s,
        },
        "memory_kb": {
            "rss_before_build": rss_before_kb,
            "rss_after_build": rss_after_build_kb,
            "rss_peak": rss_peak_kb,
        },
        "disk_collection": {
            "qps": coll_qps,
            "recall_at_k": coll_recall,
            "latency_us": {
                "p50": coll_p[50],
                "p95": coll_p[95],
                "p99": coll_p[99],
                "min": min(coll_lat_us),
                "mean": statistics.mean(coll_lat_us),
            },
        },
        "numpy_bf": (
            None
            if np_qps is None
            else {
                "qps": np_qps,
                "recall_at_k": statistics.mean(np_recalls) if np_recalls else None,
                "latency_us": {
                    "p50": np_p[50],
                    "p95": np_p[95],
                    "p99": np_p[99],
                    "min": min(np_lat_us),
                    "mean": statistics.mean(np_lat_us),
                },
            }
        ),
        "hardware": hw,
    }
    json_path.write_text(json.dumps(json_payload, indent=2))

    print()
    print("=== GIST 1M summary ===")
    print(f"  build:       {build_s:.1f}s ({n / build_s:.0f} rows/s), {n_segments} segments")
    print(f"  RSS peak:    {rss_peak_kb / 1024:.0f} MiB")
    print(
        f"  DC search:   QPS={coll_qps:.1f}, p50={coll_p[50]:.0f}us, "
        f"p95={coll_p[95]:.0f}us, p99={coll_p[99]:.0f}us, recall@{args.k}={coll_recall:.4f}"
    )
    if np_qps is not None:
        print(
            f"  numpy bf:    QPS={np_qps:.1f}, p50={np_p[50]:.0f}us, "
            f"p95={np_p[95]:.0f}us, p99={np_p[99]:.0f}us, "
            f"recall@{args.k}={statistics.mean(np_recalls):.4f}"
        )
        print(f"  ratio (DC/np): QPS={coll_qps / np_qps:.2f}x, p50={coll_p[50] / np_p[50]:.2f}x")
    print(f"\nreport: {md_path}")
    print(f"raw:    {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
