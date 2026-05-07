#!/usr/bin/env python3
"""LASER cross-bench step2: build AlayaLite Vamana graph.

Run under the AlayaLite venv (or anywhere ``alayalite.vamana`` is importable)::

    python step2_build_vamana_alayalite.py --config configs/gist1m.toml --vamana-tag alayalite_l100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CrossBenchConfig  # noqa: E402  pylint: disable=wrong-import-position

# Build-time DRAM budget for AlayaLite's Vamana builder (separate from search-time
# DRAM cache budget, which is config.bench.dram_budget_gb). 32 GB is sufficient
# for 1M-scale graphs; bump if you run on >10M-scale datasets.
_VAMANA_BUILD_DRAM_GB = 32.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--vamana-tag",
        required=True,
        help="Tag from config.vamana_sources where builder == 'alayalite'",
    )
    args = p.parse_args()

    cfg = CrossBenchConfig.from_toml(args.config)
    source = cfg.find_vamana(args.vamana_tag)
    if source.builder != "alayalite":
        raise ValueError(
            f"vamana source {source.tag!r} has builder={source.builder!r}, expected 'alayalite'. "
            "Use step1 for builder='diskann'."
        )

    base_fbin = cfg.dataset.base_fbin(cfg.paths.data_dir)
    if not base_fbin.is_file():
        raise FileNotFoundError(f"dataset base fbin not found: {base_fbin}")

    from alayalite import vamana  # pylint: disable=import-outside-toplevel

    out_dir = cfg.vamana_dir(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.vamana_path(source)

    print(
        f"[alayalite-vamana] tag={source.tag} R={cfg.build.R} L={source.L} "
        f"alpha={cfg.build.alpha} threads={cfg.bench.build_threads} seed={cfg.bench.seed} -> {out_path}"
    )
    t0 = time.perf_counter()
    vamana.build_index(
        data_path=str(base_fbin),
        output_path=str(out_path),
        R=cfg.build.R,
        L=source.L,
        alpha=cfg.build.alpha,
        seed=cfg.bench.seed,
        num_threads=cfg.bench.build_threads,
        dram_budget_gb=_VAMANA_BUILD_DRAM_GB,
    )
    print(f"[alayalite-vamana] done in {time.perf_counter() - t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
