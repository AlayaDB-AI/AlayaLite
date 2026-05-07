#!/usr/bin/env python3
"""
Full pipeline with original Laser library:
  PCA -> Medoid -> Index build -> EF sweep search -> CSV

Run twice (once per Vamana source):

    # DiskANN Vamana
    numactl --interleave=all \\
        /md1/huangliang/alaya-dev/Laser/.venv/bin/python \\
        /md1/huangliang/alaya-dev/AlayaLite/experiments/laser_cross_bench/step3_bench_original_laser.py \\
        --vamana-path /md1/huangliang/alaya-dev/tmp/laser_cross_bench/vamana_diskann/gist_vamana.index \\
        --tag diskann

    # AlayaLite Vamana
    numactl --interleave=all \\
        /md1/huangliang/alaya-dev/Laser/.venv/bin/python \\
        /md1/huangliang/alaya-dev/AlayaLite/experiments/laser_cross_bench/step3_bench_original_laser.py \\
        --vamana-path /md1/huangliang/alaya-dev/tmp/laser_cross_bench/vamana_alayalite/gist_vamana.index \\
        --tag lite
"""

from __future__ import annotations

import argparse
import csv
import struct
import time
from pathlib import Path

import numpy as np

DATA = Path("/md1/huangliang/alaya-dev/data/gist1m")
TMP = Path("/md1/huangliang/alaya-dev/tmp/laser_cross_bench")

R = 64
MAIN_DIM = 256
EF_INDEXING = 200
BUILD_THREADS = 48
K = 10
BEAM = 16
WARMUP = 10
RUNS = 30
DRAM_BUDGET = 1.0
PCA_SEED = 42
MEDOID_SEED = 42
ROTATOR_SEED = 0
EP_NUM = 300
EFS = [80, 90, 100, 110, 130, 150, 200, 250, 300, 400, 500]


def _read_fbin_shape(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
    return int(n), int(d)


def step_pca(base_fbin: str, pca_base_path: str, pca_params_path: str) -> None:
    from laser.pca import (
        fit_incremental_pca,
        pca_transform_and_save,
        sample_vectors_from_fbin,
        save_pca_params,
    )

    print("[pca] sampling and fitting...")
    t0 = time.perf_counter()
    vectors, sample_vecs = sample_vectors_from_fbin(base_fbin, seed=PCA_SEED)
    _, d = vectors.shape
    pca = fit_incremental_pca(sample_vecs, n_components=d)
    save_pca_params(pca, pca_params_path)
    pca_transform_and_save(vectors, pca, pca_base_path)
    print(f"[pca] done in {time.perf_counter() - t0:.1f}s -> {pca_base_path}")


def step_medoid(pca_base: str, idx_path: str, vec_path: str) -> None:
    from laser.medoid import generate_and_save_medoids

    print(f"[medoid] generating {EP_NUM} entry points...")
    t0 = time.perf_counter()
    generate_and_save_medoids(pca_base, idx_path, vec_path, EP_NUM, seed=MEDOID_SEED)
    print(f"[medoid] done in {time.perf_counter() - t0:.1f}s")


def step_index(pca_base: str, vamana_path: str, index_prefix: str) -> tuple[int, int]:
    import laser
    from laser.io import read_fbin

    base = read_fbin(pca_base, use_mmap=True)
    N, D = base.shape
    print(f"[index] building R{R}_MD{MAIN_DIM} N={N:,} D={D} threads={BUILD_THREADS}...")
    t0 = time.perf_counter()
    index = laser.Index(
        index_type="QG",
        metric="l2",
        num_elements=N,
        main_dimension=MAIN_DIM,
        dimension=D,
        degree_bound=R,
        rotator_seed=ROTATOR_SEED,
        rotator_dump_path="",
    )
    index.build_index(vamana_path, index_prefix, EF=EF_INDEXING, num_thread=BUILD_THREADS)
    print(f"[index] done in {time.perf_counter() - t0:.1f}s")
    return N, D


def step_search(index_prefix: str, N: int, D: int, out_csv: Path) -> None:
    import laser
    from laser.io import read_fbin, read_ibin

    query = np.asarray(read_fbin(str(DATA / "gist_query.fbin")), dtype=np.float32)
    gt = np.asarray(read_ibin(str(DATA / "gist_gt.ibin")), dtype=np.int32)
    NQ = query.shape[0]

    index = laser.Index(
        index_type="QG",
        metric="l2",
        num_elements=N,
        main_dimension=MAIN_DIM,
        dimension=D,
        degree_bound=R,
    )
    index.load(index_prefix, DRAM_BUDGET)
    print(f"[search] loaded index; NQ={NQ:,} K={K} beam={BEAM} warmup={WARMUP} runs={RUNS}")

    rows: list[dict] = []
    for ef in EFS:
        index.set_params(ef_search=ef, num_threads=1, beam_width=BEAM)

        for _ in range(WARMUP):
            for i in range(NQ):
                index.search(query[i], K)

        total_time = 0.0
        latencies_us: list[float] = []
        results: list = []
        for _ in range(RUNS):
            results = []
            for i in range(NQ):
                t0 = time.perf_counter()
                pred = index.search(query[i], K)
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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ef", "qps", "recall_at_10", "mean_lat_us", "p99_9_lat_us"])
        w.writeheader()
        w.writerows(rows)
    print(f"[search] written -> {out_csv}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vamana-path", required=True, type=Path, help="Pre-built Vamana .index file (DiskANN format)")
    p.add_argument("--tag", required=True, help="Label for the Vamana source (used in output path)")
    args = p.parse_args()

    artifacts = TMP / f"orig_{args.tag}"
    artifacts.mkdir(parents=True, exist_ok=True)

    pca_base = str(artifacts / "dsqg_gist_pca_base.fbin")
    pca_params = str(artifacts / "dsqg_gist_pca.bin")
    medoid_idx = str(artifacts / "dsqg_gist_medoids_indices")
    medoid_vecs = str(artifacts / "dsqg_gist_medoids")
    index_prefix = str(artifacts / "dsqg_gist")

    step_pca(str(DATA / "gist_base.fbin"), pca_base, pca_params)
    step_medoid(pca_base, medoid_idx, medoid_vecs)
    N, D = step_index(pca_base, str(args.vamana_path), index_prefix)

    out_csv = TMP / "results" / f"orig_{args.tag}.csv"
    step_search(index_prefix, N, D, out_csv)


if __name__ == "__main__":
    main()
