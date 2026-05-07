#!/usr/bin/env python3
"""
Full pipeline with AlayaLite Laser:
  (pre-place Vamana) -> Index.fit -> EF sweep search -> CSV

Run twice (once per Vamana source):

    # DiskANN Vamana
    numactl --interleave=all \\
        uv run python experiments/laser_cross_bench/step4_bench_alayalite_laser.py \\
        --vamana-path /md1/huangliang/alaya-dev/tmp/laser_cross_bench/vamana_diskann/gist_vamana.index \\
        --tag diskann

    # AlayaLite Vamana
    numactl --interleave=all \\
        uv run python experiments/laser_cross_bench/step4_bench_alayalite_laser.py \\
        --vamana-path /md1/huangliang/alaya-dev/tmp/laser_cross_bench/vamana_alayalite/gist_vamana.index \\
        --tag lite
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np

DATA = Path("/md1/huangliang/alaya-dev/data/gist1m")
TMP = Path("/md1/huangliang/alaya-dev/tmp/laser_cross_bench")

R = 64
MAIN_DIM = 256
L_VAMANA = 100  # AlayaLite vamana L (skipped if vamana pre-placed)
ALPHA = 1.2
EF_INDEXING = 200
BUILD_THREADS = 48
K = 10
BEAM = 16
WARMUP = 10
RUNS = 30
DRAM_BUDGET = 1.0
SEED = 42
EP_NUM = 300
EFS = [80, 90, 100, 110, 130, 150, 200, 250, 300, 400, 500]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vamana-path", required=True, type=Path, help="Pre-built Vamana .index file (DiskANN format)")
    p.add_argument("--tag", required=True, choices=["diskann", "lite"], help="Label for the Vamana source")
    args = p.parse_args()

    from alayalite import laser
    from alayalite.laser._io import read_fbin, read_ibin

    output_dir = TMP / f"lite_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-place vamana so Index.fit(skip_existing=True) skips its internal build.
    # Index.fit with name="laser" expects the vamana at "{output_dir}/laser_vamana_graph.index".
    vamana_target = output_dir / "laser_vamana_graph.index"
    print(f"[prep] copying vamana -> {vamana_target}")
    shutil.copyfile(args.vamana_path, vamana_target)

    print(f"[fit] building LASER R{R}_MD{MAIN_DIM} threads={BUILD_THREADS}...")
    t0 = time.perf_counter()
    idx = laser.Index.fit(
        str(DATA / "gist_base.fbin"),
        output_dir=output_dir,
        name="laser",
        metric="l2",
        main_dim=MAIN_DIM,
        R=R,
        L=L_VAMANA,
        alpha=ALPHA,
        ef_indexing=EF_INDEXING,
        beam_width=BEAM,
        num_threads=BUILD_THREADS,
        ep_num=EP_NUM,
        seed=SEED,
        dram_budget_gb=DRAM_BUDGET,
        disable_medoid=False,
        skip_existing=True,
        auto_load=True,
    )
    print(f"[fit] done in {time.perf_counter() - t0:.1f}s")

    query = np.ascontiguousarray(read_fbin(str(DATA / "gist_query.fbin"), use_mmap=False), dtype=np.float32)
    gt = np.asarray(read_ibin(str(DATA / "gist_gt.ibin")), dtype=np.int32)
    NQ = query.shape[0]
    print(f"[search] NQ={NQ:,} K={K} beam={BEAM} warmup={WARMUP} runs={RUNS}")

    rows: list[dict] = []
    for ef in EFS:
        idx.set_params(ef_search=ef, num_threads=1, beam_width=BEAM)

        for _ in range(WARMUP):
            for i in range(NQ):
                idx.search(query[i], K)

        total_time = 0.0
        latencies_us: list[float] = []
        results: list = []
        for _ in range(RUNS):
            results = []
            for i in range(NQ):
                t0 = time.perf_counter()
                pred = idx.search(query[i], K)
                t1 = time.perf_counter()
                results.append(pred)
                latencies_us.append((t1 - t0) * 1e6)
                total_time += t1 - t0

        correct = sum(1 for i in range(NQ) for j in range(K) if gt[i][j] in set(results[i]))
        recall = correct / (NQ * K)
        qps = NQ * RUNS / total_time
        mean_lat = float(np.mean(latencies_us))
        p99_lat = float(np.percentile(latencies_us, 99.9))

        rows.append(
            {
                "ef": ef,
                "qps": round(qps, 2),
                "recall_at_10": round(recall, 6),
                "mean_lat_us": round(mean_lat, 2),
                "p99_9_lat_us": round(p99_lat, 2),
            }
        )
        print(f"  EF={ef:>4}  QPS={qps:>8.1f}  Recall={recall:.4f}  mean={mean_lat:.0f}us  p99.9={p99_lat:.0f}us")

    out_csv = TMP / "results" / f"lite_{args.tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ef", "qps", "recall_at_10", "mean_lat_us", "p99_9_lat_us"])
        w.writeheader()
        w.writerows(rows)
    print(f"[search] written -> {out_csv}")


if __name__ == "__main__":
    main()
