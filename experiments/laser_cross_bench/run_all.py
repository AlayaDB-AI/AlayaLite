#!/usr/bin/env python3
"""LASER cross-bench top-level driver.

Iterates ``vamana_sources × {orig laser, lite laser}`` from a config and runs
the full pipeline (step1/step2 build, step3/step4 bench, step5 compare,
plot_recall_qps), respecting ``laser_venv`` for the orig pipeline.

Skips steps whose artifacts already exist (override with --force-*).

Run from anywhere::

    python run_all.py --config configs/gist1m.toml

    # only orig laser side, only the diskann vamana
    python run_all.py --config configs/gist1m.toml --laser-only orig --tags diskann

    # rebuild everything from scratch
    python run_all.py --config configs/gist1m.toml --force-vamana --force-bench
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CrossBenchConfig  # noqa: E402  pylint: disable=wrong-import-position

THIS_DIR = Path(__file__).resolve().parent


def _run(python: Path | str, script: str, args: list[str], *, use_numactl: bool) -> None:
    cmd: list[str] = []
    if use_numactl:
        cmd += ["numactl", "--interleave=all"]
    cmd += [str(python), str(THIS_DIR / script), *args]
    print(">>>", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Subset of vamana_source tags to drive (default: all)",
    )
    p.add_argument(
        "--laser-only",
        choices=("orig", "lite"),
        default=None,
        help="Run only one laser side (default: both)",
    )
    p.add_argument("--force-vamana", action="store_true", help="Rebuild Vamana graphs even if present")
    p.add_argument("--force-bench", action="store_true", help="Rerun bench cells even if CSV present")
    p.add_argument("--skip-plot", action="store_true", help="Skip step5 + plot_recall_qps")
    args = p.parse_args()

    cfg = CrossBenchConfig.from_toml(args.config)
    cfg_path = str(args.config.resolve())
    use_numactl = cfg.bench.numactl

    sources = list(cfg.vamana_sources)
    if args.tags is not None:
        wanted = set(args.tags)
        sources = [vs for vs in sources if vs.tag in wanted]
        unknown = wanted - {vs.tag for vs in cfg.vamana_sources}
        if unknown:
            raise ValueError(f"unknown vamana tags: {sorted(unknown)}")
    if not sources:
        raise ValueError("no vamana sources selected")

    lasers = (args.laser_only,) if args.laser_only else ("orig", "lite")

    current_python = Path(sys.executable)
    laser_python = {
        "orig": cfg.paths.require_laser_venv() if "orig" in lasers else current_python,
        "lite": current_python,
    }
    laser_script = {
        "orig": "step3_bench_original_laser.py",
        "lite": "step4_bench_alayalite_laser.py",
    }

    # ─── step1/step2: build Vamana graphs ────────────────────────────────────
    for source in sources:
        out_path = cfg.vamana_path(source)
        if out_path.is_file() and not args.force_vamana:
            print(f"[vamana:skip] {source.tag} already built -> {out_path}")
            continue
        if source.builder == "diskann":
            _run(
                current_python,
                "step1_build_vamana_diskann.py",
                ["--config", cfg_path, "--vamana-tag", source.tag],
                use_numactl=False,  # step1 wraps numactl internally around the diskann binary
            )
        else:
            _run(
                current_python,
                "step2_build_vamana_alayalite.py",
                ["--config", cfg_path, "--vamana-tag", source.tag],
                use_numactl=use_numactl,
            )

    # ─── step3/step4: bench cells ────────────────────────────────────────────
    for source in sources:
        for laser in lasers:
            csv_path = cfg.cell_csv(laser, source.tag)
            if csv_path.is_file() and not args.force_bench:
                print(f"[bench:skip] {laser}+{source.tag} csv exists -> {csv_path}")
                continue
            _run(
                laser_python[laser],
                laser_script[laser],
                ["--config", cfg_path, "--vamana-tag", source.tag],
                use_numactl=use_numactl,
            )

    if args.skip_plot:
        return

    # ─── step5 + plot ────────────────────────────────────────────────────────
    _run(current_python, "step5_compare.py", ["--config", cfg_path], use_numactl=False)
    _run(current_python, "plot_recall_qps.py", ["--config", cfg_path], use_numactl=False)


if __name__ == "__main__":
    main()
