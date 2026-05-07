#!/usr/bin/env python3
"""LASER cross-bench step1: build DiskANN Vamana graph.

Thin wrapper around DiskANN's ``build_memory_index`` binary
(``config.paths.diskann_binary``). Run under any Python venv; the binary
itself is independent::

    python step1_build_vamana_diskann.py --config configs/gist1m.toml --vamana-tag diskann
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CrossBenchConfig  # noqa: E402  pylint: disable=wrong-import-position


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--vamana-tag",
        required=True,
        help="Tag from config.vamana_sources where builder == 'diskann'",
    )
    args = p.parse_args()

    cfg = CrossBenchConfig.from_toml(args.config)
    source = cfg.find_vamana(args.vamana_tag)
    if source.builder != "diskann":
        raise ValueError(
            f"vamana source {source.tag!r} has builder={source.builder!r}, expected 'diskann'. "
            "Use step2 for builder='alayalite'."
        )

    diskann = cfg.paths.require_diskann_binary()
    if not diskann.is_file():
        raise FileNotFoundError(f"DiskANN binary not found: {diskann}")

    base_fbin = cfg.dataset.base_fbin(cfg.paths.data_dir)
    if not base_fbin.is_file():
        raise FileNotFoundError(f"dataset base fbin not found: {base_fbin}")

    out_dir = cfg.vamana_dir(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.vamana_path(source)

    cmd: list[str] = []
    if cfg.bench.numactl:
        cmd += ["numactl", "--interleave=all"]
    cmd += [
        str(diskann),
        "--data_type",
        "float",
        "--dist_fn",
        "l2",
        "--data_path",
        str(base_fbin),
        "--index_path_prefix",
        str(out_path),
        "-R",
        str(cfg.build.R),
        "-L",
        str(source.L),
        "--alpha",
        str(cfg.build.alpha),
        "--num_threads",
        str(cfg.bench.build_threads),
    ]
    print(
        f"[diskann-vamana] tag={source.tag} R={cfg.build.R} L={source.L} "
        f"alpha={cfg.build.alpha} threads={cfg.bench.build_threads} -> {out_path}"
    )
    print("[diskann-vamana] cmd:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)
    print(f"[diskann-vamana] done -> {out_path}")


if __name__ == "__main__":
    main()
