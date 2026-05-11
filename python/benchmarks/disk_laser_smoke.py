#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Compatibility wrapper around `python -m alayalite.bench.disk_collection`.

Preserves the legacy stdout contract from
`openspec/specs/disk-laser-python-binding/spec.md` (lines matching
`recall@k:`, `p50_ms:`, `p95_ms:`, `qps:`, `import_seconds:`,
`segment_bytes:`, plus the `disk_laser not available on this build`
graceful-skip line) by post-processing the harness's JSON output.

See openspec change `disk-collection-benchmark-suite` for full context.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _emit_legacy_lines(summary: dict) -> None:
    """Print the lines required by disk-laser-python-binding spec scenarios."""
    raws = summary.get("raws") or []
    if not raws:
        return
    raw = raws[0]
    status = raw.get("status", "ok")
    results = raw.get("results", {})
    params = raw.get("params", {})

    if status == "skipped":
        # Legacy scenario "Smoke benchmark gracefully exits on unsupported build"
        # requires the literal line below.
        print("disk_laser not available on this build")
        return

    latency = results.get("latency_us") or {}
    top_k = params.get("top_k", "?")
    recall_value = results.get(f"recall_at_{top_k}")
    if recall_value is None:
        recall_value = results.get("recall_at_10")
    print(f"recall@{top_k}: {recall_value if recall_value is not None else 'N/A'}")
    p50 = latency.get("p50")
    p95 = latency.get("p95")
    print(f"p50_ms: {p50 / 1000 if p50 is not None else 'N/A'}")
    print(f"p95_ms: {p95 / 1000 if p95 is not None else 'N/A'}")
    qps = results.get("qps")
    print(f"qps: {qps if qps is not None else 'N/A'}")
    build_s = results.get("build_wall_s")
    print(f"import_seconds: {build_s if build_s is not None else 'N/A'}")
    seg_bytes = results.get("on_disk_bytes")
    print(f"segment_bytes: {seg_bytes if seg_bytes is not None else 'N/A'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", required=True)
    for name, default in (
        ("--n", "10000"),
        ("--dim", "128"),
        ("--queries", "100"),
        ("--k", "10"),
        ("--ef", "128"),
        ("--beam-width", "4"),
        ("--seed", "12345"),
        ("--out", None),
    ):
        parser.add_argument(name, default=default)
    for name in ("--vectors", "--ground-truth", "--queries-path"):
        parser.add_argument(name, default=None)
    args = parser.parse_args()

    # Use a temp dir if --out not given so the wrapper stays self-contained
    # and the JSON we read for legacy-line emission is unambiguous.
    out_root_ctx = tempfile.TemporaryDirectory(prefix="alayalite_disk_laser_smoke_") if args.out is None else None
    out_root = Path(args.out) if args.out is not None else Path(out_root_ctx.name)
    run_id = "smoke"
    cmd = [
        sys.executable,
        "-m",
        "alayalite.bench.disk_collection",
        "--engine",
        "disk_laser",
        "--dataset",
        "laser_files",
        "--laser-src-dir",
        args.src_dir,
        "--n",
        args.n,
        "--dim",
        args.dim,
        "--queries",
        args.queries,
        "--k",
        args.k,
        "--ef",
        args.ef,
        "--beam-width",
        args.beam_width,
        "--seed",
        args.seed,
        "--out",
        str(out_root),
        "--run-id",
        run_id,
    ]
    for flag, value in (
        ("--vectors", args.vectors),
        ("--ground-truth", args.ground_truth),
        ("--queries-path", args.queries_path),
    ):
        if value is not None:
            cmd.extend([flag, value])
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            summary_path = out_root / run_id / "summary.json"
            if summary_path.is_file():
                _emit_legacy_lines(json.loads(summary_path.read_text(encoding="utf-8")))
        return result.returncode
    finally:
        if out_root_ctx is not None:
            out_root_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
