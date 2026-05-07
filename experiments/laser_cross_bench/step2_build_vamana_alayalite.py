#!/usr/bin/env python3
"""Build AlayaLite Vamana graph for GIST-1M.

Run with (numactl --interleave=all pins memory across NUMA nodes, matching
DiskANN build policy; 48 threads keeps build time under 5 minutes):

    numactl --interleave=all \\
        uv run python experiments/laser_cross_bench/step2_build_vamana_alayalite.py
"""

from __future__ import annotations

import time
from pathlib import Path

DATA = Path("/md1/huangliang/alaya-dev/data/gist1m")
TMP = Path("/md1/huangliang/alaya-dev/tmp/laser_cross_bench")

R = 64
L = 100
ALPHA = 1.2
THREADS = 48
SEED = 42
DRAM_GB = 32.0


def main() -> None:
    from alayalite import vamana

    out_dir = TMP / "vamana_alayalite"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gist_vamana.index"

    print(f"[alayalite-vamana] building R={R} L={L} alpha={ALPHA} threads={THREADS} seed={SEED} → {out_path}")
    t0 = time.perf_counter()
    vamana.build_index(
        data_path=str(DATA / "gist_base.fbin"),
        output_path=str(out_path),
        R=R,
        L=L,
        alpha=ALPHA,
        seed=SEED,
        num_threads=THREADS,
        dram_budget_gb=DRAM_GB,
    )
    elapsed = time.perf_counter() - t0
    print(f"[alayalite-vamana] done in {elapsed:.1f}s → {out_path}")


if __name__ == "__main__":
    main()
