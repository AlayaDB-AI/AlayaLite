# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Smoke perf check: DiskCollection.search vs numpy brute-force at 10k x 128.

Goal: confirm we are in the same ballpark as a tight numpy reference. NOT
a benchmark — small dataset, single-threaded, no -march tuning. Output is
a markdown summary written into results/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from alayalite import DiskCollection, MetricType


def hw_provenance() -> dict:
    """Capture host info that any reproducer must record."""
    info: dict = {
        "uname": platform.uname()._asdict(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
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
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = line.split(":", 1)[1].split()
                    info["simd_flags"] = sorted(f for f in flags if f.startswith(("sse4", "avx")))
                    break
    except OSError:
        pass
    return info


def percentiles(values: list[float], qs=(50, 95, 99)) -> dict[int, float]:
    s = sorted(values)
    out: dict[int, float] = {}
    for q in qs:
        idx = max(0, min(len(s) - 1, int(len(s) * q / 100)))
        out[q] = s[idx]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", choices=["L2", "IP", "COS"], default="L2")
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    metric = {"L2": MetricType.L2, "IP": MetricType.IP, "COS": MetricType.COS}[args.metric]

    rng = np.random.default_rng(args.seed)
    vectors = rng.standard_normal((args.n, args.dim)).astype(np.float32)
    ids = np.arange(args.n, dtype=np.uint64)
    queries = rng.standard_normal((args.queries + args.warmup, args.dim)).astype(np.float32)

    dataset_sha = hashlib.sha256(b"".join([vectors.tobytes(), ids.tobytes(), queries.tobytes()])).hexdigest()[:16]

    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ---- Phase A: DiskCollection ---------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "smoke_coll")

        t0 = time.perf_counter()
        col = DiskCollection(path=path, dim=args.dim, metric=metric, index_type="disk_flat")
        col.add(vectors, ids)
        col.flush()
        build_s = time.perf_counter() - t0
        rss_after_build_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Warmup.
        for i in range(args.warmup):
            col.search(queries[i], k=args.k)

        # Timed loop.
        coll_lat_us: list[float] = []
        t_start = time.perf_counter()
        for i in range(args.warmup, args.warmup + args.queries):
            t = time.perf_counter()
            col.search(queries[i], k=args.k)
            coll_lat_us.append((time.perf_counter() - t) * 1e6)
        coll_total_s = time.perf_counter() - t_start
        coll_qps = args.queries / coll_total_s
        coll_p = percentiles(coll_lat_us)

    # ---- Phase B: numpy brute-force (reference) ------------------------
    # Same warmup pattern.
    for i in range(args.warmup):
        diffs = vectors - queries[i]
        np.argpartition((diffs * diffs).sum(axis=1), args.k)[: args.k]

    np_lat_us: list[float] = []
    t_start = time.perf_counter()
    for i in range(args.warmup, args.warmup + args.queries):
        t = time.perf_counter()
        diffs = vectors - queries[i]
        np.argpartition((diffs * diffs).sum(axis=1), args.k)[: args.k]
        np_lat_us.append((time.perf_counter() - t) * 1e6)
    np_total_s = time.perf_counter() - t_start
    np_qps = args.queries / np_total_s
    np_p = percentiles(np_lat_us)

    # ---- Phase C: assemble report --------------------------------------
    rss_peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    md_path = out_dir / f"disk_collection_smoke_{args.metric.lower()}_{ts}.md"
    json_path = out_dir / f"disk_collection_smoke_{args.metric.lower()}_{ts}.json"

    hw = hw_provenance()
    md_lines = [
        "# DiskCollection smoke perf",
        "",
        f"- timestamp: {ts}",
        f"- dataset: n={args.n}, dim={args.dim}, metric={args.metric}, seed={args.seed}",
        f"- queries: {args.queries} timed (+{args.warmup} warmup), k={args.k}",
        f"- dataset_sha (first 16 hex of vectors+ids+queries): `{dataset_sha}`",
        f"- cpu: {hw.get('cpu_model', 'unknown')}",
        f"- cpu_count: {hw.get('cpu_count')}",
        f"- mem_total_kb: {hw.get('mem_total_kb')}",
        f"- simd_flags: {' '.join(hw.get('simd_flags', []))}",
        f"- python: {hw.get('python')} on {hw['uname']['system']} {hw['uname']['release']}",
        "- compiler flags: -Ofast -DNDEBUG (release build)",
        "",
        "## Results",
        "",
        "| | DiskCollection | numpy bf | ratio |",
        "|---|---:|---:|---:|",
        f"| build/flush time (ms) | {build_s * 1000:.1f} | n/a | n/a |",
        f"| QPS | {coll_qps:.0f} | {np_qps:.0f} | {coll_qps / np_qps:.2f}x |",
        f"| p50 latency (us) | {coll_p[50]:.1f} | {np_p[50]:.1f} | {coll_p[50] / np_p[50]:.2f}x |",
        f"| p95 latency (us) | {coll_p[95]:.1f} | {np_p[95]:.1f} | {coll_p[95] / np_p[95]:.2f}x |",
        f"| p99 latency (us) | {coll_p[99]:.1f} | {np_p[99]:.1f} | {coll_p[99] / np_p[99]:.2f}x |",
        f"| min latency (us) | {min(coll_lat_us):.1f} | {min(np_lat_us):.1f} | "
        f"{min(coll_lat_us) / min(np_lat_us):.2f}x |",
        f"| mean latency (us) | {statistics.mean(coll_lat_us):.1f} | "
        f"{statistics.mean(np_lat_us):.1f} | "
        f"{statistics.mean(coll_lat_us) / statistics.mean(np_lat_us):.2f}x |",
        "",
        "## Memory",
        "",
        f"- RSS before build: {rss_before_kb} KB",
        f"- RSS after build/flush: {rss_after_build_kb} KB",
        f"- RSS peak (process lifetime): {rss_peak_kb} KB",
        "",
        "## Notes",
        "",
        "- Smoke check, not a benchmark. Single-threaded, no `-march=native` tuning.",
        "- Both phases reuse the same query buffer and warmup count.",
        "- numpy reference uses `argpartition` (top-k without full sort), so it is "
        "intentionally fast and not directly comparable to our heap-based path.",
        "- Recall is 1.0 by construction in both phases (exact brute force).",
    ]
    md_path.write_text("\n".join(md_lines) + "\n")
    json_path.write_text(
        json.dumps(
            {
                "timestamp": ts,
                "dataset": {
                    "n": args.n,
                    "dim": args.dim,
                    "metric": args.metric,
                    "seed": args.seed,
                    "sha16": dataset_sha,
                },
                "queries": {"timed": args.queries, "warmup": args.warmup, "k": args.k},
                "disk_collection": {
                    "build_ms": build_s * 1000,
                    "qps": coll_qps,
                    "latency_us": {
                        "p50": coll_p[50],
                        "p95": coll_p[95],
                        "p99": coll_p[99],
                        "min": min(coll_lat_us),
                        "mean": statistics.mean(coll_lat_us),
                    },
                },
                "numpy_bf": {
                    "qps": np_qps,
                    "latency_us": {
                        "p50": np_p[50],
                        "p95": np_p[95],
                        "p99": np_p[99],
                        "min": min(np_lat_us),
                        "mean": statistics.mean(np_lat_us),
                    },
                },
                "memory_kb": {
                    "rss_before_build": rss_before_kb,
                    "rss_after_build": rss_after_build_kb,
                    "rss_peak": rss_peak_kb,
                },
                "hardware": hw,
            },
            indent=2,
        )
    )

    print(f"\n=== smoke perf summary ({args.metric}) ===")
    print(
        f"DiskCollection: build={build_s * 1000:.1f}ms, QPS={coll_qps:.0f}, "
        f"p50={coll_p[50]:.1f}us, p95={coll_p[95]:.1f}us, p99={coll_p[99]:.1f}us"
    )
    print(f"numpy bf:       QPS={np_qps:.0f}, p50={np_p[50]:.1f}us, p95={np_p[95]:.1f}us, p99={np_p[99]:.1f}us")
    print(f"ratio (DC/np):  QPS={coll_qps / np_qps:.2f}x, p50={coll_p[50] / np_p[50]:.2f}x")
    print(f"\nreport: {md_path}")
    print(f"raw:    {json_path}")


if __name__ == "__main__":
    sys.exit(main())
