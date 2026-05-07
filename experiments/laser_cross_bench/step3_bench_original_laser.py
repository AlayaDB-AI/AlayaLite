#!/usr/bin/env python3
"""LASER cross-bench step3: original Laser PCA -> Medoid -> Index -> EF sweep.

Imports the upstream ``laser`` package, so it must run under
``config.paths.laser_venv``::

    /path/to/Laser/.venv/bin/python step3_bench_original_laser.py \\
        --config configs/gist1m.toml \\
        --vamana-tag diskann

Outputs:
  * Build artifacts under ``<tmp_dir>/orig_<vamana_tag>/dsqg_<prefix>_*``
  * Search results CSV at ``<tmp_dir>/results/orig_<vamana_tag>.csv``
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# Local sibling import; Laser venv has no _config in site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CrossBenchConfig  # noqa: E402  pylint: disable=wrong-import-position


def step_pca(cfg: CrossBenchConfig, base_fbin: Path, pca_base_path: Path, pca_params_path: Path) -> None:
    from laser.pca import (  # pylint: disable=import-outside-toplevel,import-error
        fit_incremental_pca,
        pca_transform_and_save,
        sample_vectors_from_fbin,
        save_pca_params,
    )

    print("[pca] sampling and fitting...")
    t0 = time.perf_counter()
    vectors, sample_vecs = sample_vectors_from_fbin(str(base_fbin), seed=cfg.bench.seed)
    _, d = vectors.shape
    pca = fit_incremental_pca(sample_vecs, n_components=d)
    save_pca_params(pca, str(pca_params_path))
    pca_transform_and_save(vectors, pca, str(pca_base_path))
    print(f"[pca] done in {time.perf_counter() - t0:.1f}s -> {pca_base_path}")


def step_medoid(cfg: CrossBenchConfig, pca_base: Path, idx_path: Path, vec_path: Path) -> None:
    from laser.medoid import generate_and_save_medoids  # pylint: disable=import-outside-toplevel,import-error

    print(f"[medoid] generating {cfg.build.ep_num} entry points...")
    t0 = time.perf_counter()
    generate_and_save_medoids(
        str(pca_base),
        str(idx_path),
        str(vec_path),
        cfg.build.ep_num,
        seed=cfg.bench.seed,
    )
    print(f"[medoid] done in {time.perf_counter() - t0:.1f}s")


def step_index(cfg: CrossBenchConfig, pca_base: Path, vamana_path: Path, index_prefix: str) -> tuple[int, int]:
    import laser  # pylint: disable=import-outside-toplevel,import-error
    from laser.io import read_fbin  # pylint: disable=import-outside-toplevel,import-error

    base = read_fbin(str(pca_base), use_mmap=True)
    n, d = base.shape
    print(f"[index] building R{cfg.build.R}_MD{cfg.build.main_dim} N={n:,} D={d} threads={cfg.bench.build_threads}...")
    t0 = time.perf_counter()
    index = laser.Index(
        index_type="QG",
        metric="l2",
        num_elements=n,
        main_dimension=cfg.build.main_dim,
        dimension=d,
        degree_bound=cfg.build.R,
        rotator_seed=cfg.bench.seed,
        rotator_dump_path="",
    )
    index.build_index(
        str(vamana_path),
        index_prefix,
        EF=cfg.build.ef_indexing,
        num_thread=cfg.bench.build_threads,
    )
    print(f"[index] done in {time.perf_counter() - t0:.1f}s")
    return n, d


def step_search(cfg: CrossBenchConfig, index_prefix: str, n: int, d: int, out_csv: Path) -> None:
    import laser  # pylint: disable=import-outside-toplevel,import-error
    from laser.io import read_fbin, read_ibin  # pylint: disable=import-outside-toplevel,import-error

    query = np.asarray(read_fbin(str(cfg.dataset.query_fbin(cfg.paths.data_dir))), dtype=np.float32)
    gt = np.asarray(read_ibin(str(cfg.dataset.gt_ibin(cfg.paths.data_dir))), dtype=np.int32)
    nq = query.shape[0]

    index = laser.Index(
        index_type="QG",
        metric="l2",
        num_elements=n,
        main_dimension=cfg.build.main_dim,
        dimension=d,
        degree_bound=cfg.build.R,
    )
    index.load(index_prefix, cfg.bench.dram_budget_gb)
    print(
        f"[search] loaded index; NQ={nq:,} K={cfg.bench.k} beam={cfg.bench.beam} "
        f"warmup={cfg.bench.warmup} runs={cfg.bench.runs}"
    )

    rows: list[dict] = []
    for ef in cfg.bench.ef_sweep:
        index.set_params(ef_search=ef, num_threads=cfg.bench.search_threads, beam_width=cfg.bench.beam)

        for _ in range(cfg.bench.warmup):
            for i in range(nq):
                index.search(query[i], cfg.bench.k)

        total_time = 0.0
        latencies_us: list[float] = []
        results: list = []
        for _ in range(cfg.bench.runs):
            results = []
            for i in range(nq):
                t0 = time.perf_counter()
                pred = index.search(query[i], cfg.bench.k)
                t1 = time.perf_counter()
                results.append(pred)
                latencies_us.append((t1 - t0) * 1e6)
                total_time += t1 - t0

        correct = sum(1 for i in range(nq) for j in range(cfg.bench.k) if gt[i][j] in set(results[i]))
        recall = correct / (nq * cfg.bench.k)
        qps = nq * cfg.bench.runs / total_time
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
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ef", "qps", "recall_at_10", "mean_lat_us", "p99_9_lat_us"])
        w.writeheader()
        w.writerows(rows)
    print(f"[search] written -> {out_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="Path to cross-bench TOML config")
    p.add_argument(
        "--vamana-tag",
        required=True,
        help="Tag from config.vamana_sources (e.g. 'diskann', 'alayalite_l100')",
    )
    args = p.parse_args()

    cfg = CrossBenchConfig.from_toml(args.config)
    source = cfg.find_vamana(args.vamana_tag)
    vamana_path = cfg.vamana_path(source)
    if not vamana_path.is_file():
        raise FileNotFoundError(
            f"Vamana graph not found at {vamana_path}. Run step1/step2 (builder={source.builder!r}) first."
        )

    base_fbin = cfg.dataset.base_fbin(cfg.paths.data_dir)
    if not base_fbin.is_file():
        raise FileNotFoundError(f"dataset base fbin not found: {base_fbin}")

    cell_dir = cfg.cell_dir("orig", source.tag)
    cell_dir.mkdir(parents=True, exist_ok=True)
    pfx = cfg.cell_index_prefix("orig", source.tag)
    pca_base = Path(f"{pfx}_pca_base.fbin")
    pca_params = Path(f"{pfx}_pca.bin")
    medoid_idx = Path(f"{pfx}_medoids_indices")
    medoid_vecs = Path(f"{pfx}_medoids")

    step_pca(cfg, base_fbin, pca_base, pca_params)
    step_medoid(cfg, pca_base, medoid_idx, medoid_vecs)
    n, d = step_index(cfg, pca_base, vamana_path, pfx)
    step_search(cfg, pfx, n, d, cfg.cell_csv("orig", source.tag))


if __name__ == "__main__":
    main()
