#!/usr/bin/env python3
"""LASER cross-bench step4: AlayaLite Laser Index.fit -> EF sweep.

Run under the AlayaLite venv::

    /path/to/AlayaLite/.venv/bin/python step4_bench_alayalite_laser.py \\
        --config configs/gist1m.toml \\
        --vamana-tag diskann

Outputs:
  * Build artifacts under ``<tmp_dir>/lite_<vamana_tag>/laser_*``
  * Search results CSV at ``<tmp_dir>/results/lite_<vamana_tag>.csv``
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CrossBenchConfig  # noqa: E402  pylint: disable=wrong-import-position


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
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

    from alayalite import laser  # pylint: disable=import-outside-toplevel
    from alayalite.laser._io import read_fbin, read_ibin  # pylint: disable=import-outside-toplevel

    base_fbin = cfg.dataset.base_fbin(cfg.paths.data_dir)
    if not base_fbin.is_file():
        raise FileNotFoundError(f"dataset base fbin not found: {base_fbin}")

    output_dir = cfg.cell_dir("lite", source.tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-place the externally-built Vamana so Index.fit(skip_existing=True)
    # reuses it instead of triggering a fresh internal build.
    vamana_target = output_dir / "laser_vamana_graph.index"
    print(f"[prep] copying vamana -> {vamana_target}")
    shutil.copyfile(vamana_path, vamana_target)

    print(f"[fit] building LASER R{cfg.build.R}_MD{cfg.build.main_dim} threads={cfg.bench.build_threads}...")
    t0 = time.perf_counter()
    idx = laser.Index.fit(
        str(base_fbin),
        output_dir=output_dir,
        name="laser",
        build_params=laser.BuildParams(
            main_dim=cfg.build.main_dim,
            R=cfg.build.R,
            L=cfg.build.L,
            alpha=cfg.build.alpha,
            ef_indexing=cfg.build.ef_indexing,
            ep_num=cfg.build.ep_num,
        ),
        num_threads=cfg.bench.build_threads,
        seed=cfg.bench.seed,
        dram_budget_gb=cfg.bench.dram_budget_gb,
        skip_existing=True,
        auto_load=True,
    )
    idx.set_params(
        ef_search=cfg.build.ef_indexing,
        num_threads=cfg.bench.search_threads,
        beam_width=cfg.bench.beam,
    )
    print(f"[fit] done in {time.perf_counter() - t0:.1f}s")

    query = np.ascontiguousarray(
        read_fbin(str(cfg.dataset.query_fbin(cfg.paths.data_dir)), use_mmap=False),
        dtype=np.float32,
    )
    gt = np.asarray(read_ibin(str(cfg.dataset.gt_ibin(cfg.paths.data_dir))), dtype=np.int32)
    nq = query.shape[0]
    print(f"[search] NQ={nq:,} K={cfg.bench.k} beam={cfg.bench.beam} warmup={cfg.bench.warmup} runs={cfg.bench.runs}")

    rows: list[dict] = []
    for ef in cfg.bench.ef_sweep:
        idx.set_params(ef_search=ef, num_threads=cfg.bench.search_threads, beam_width=cfg.bench.beam)

        for _ in range(cfg.bench.warmup):
            for i in range(nq):
                idx.search(query[i], cfg.bench.k)

        total_time = 0.0
        latencies_us: list[float] = []
        results: list = []
        for _ in range(cfg.bench.runs):
            results = []
            for i in range(nq):
                t0 = time.perf_counter()
                pred = idx.search(query[i], cfg.bench.k)
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

    out_csv = cfg.cell_csv("lite", source.tag)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ef", "qps", "recall_at_10", "mean_lat_us", "p99_9_lat_us"])
        w.writeheader()
        w.writerows(rows)
    print(f"[search] written -> {out_csv}")


if __name__ == "__main__":
    main()
