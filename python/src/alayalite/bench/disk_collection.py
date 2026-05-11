# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unified benchmark harness for `alayalite.DiskCollection`.

Run with `python -m alayalite.bench.disk_collection --engine disk_flat
--dataset synth --n 10000 --queries 1000 --k 10 --metric L2`.

The JSON output uses `schema_version=1`; see OpenSpec change
`disk-collection-benchmark-suite`, design D4, for the full schema contract.
"""

# pylint: disable=inconsistent-quotes  # f-string subscript syntax requires single
#                                        quotes inside double-quoted f-strings on
#                                        Python 3.10 (.venv-py310-backup).
# pylint: disable=too-many-positional-arguments  # _iter_sweep_points has six
#                                                  unavoidable axes; design D1.

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from ._datasets import DatasetMissing, DatasetSpec, load_gist1m, load_laser_files, load_sift1m, load_synth
from ._engines import bench_disk_flat, bench_disk_laser, bench_disk_vamana, probe_disk_laser_supported
from ._metrics import render_raw_json, render_summary_json, render_summary_md, write_json
from ._provenance import collect_provenance

ENGINES = ("disk_flat", "disk_vamana", "disk_laser")
DATASETS = ("synth", "sift1m", "gist1m", "laser_files")
METRICS = ("L2", "IP", "COS")
SWEEP_TOP_K = (1, 10, 100)
SWEEP_EF = (50, 100, 200, 400)
SWEEP_BEAM_WIDTH = (1, 2, 4, 8)
SWEEP_PENDING_BYTES = (None, 1_048_576, 262_144, 65_536)


@dataclass(frozen=True)
class ParamPoint:
    top_k: int
    ef: Optional[int]
    beam_width: Optional[int]
    max_pending_bytes: Optional[int]


def _parse_max_pending_bytes(value: str) -> Optional[int]:
    if value == "default":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-pending-bytes must be positive or 'default'")
    return parsed


def _iter_sweep_points(
    engine: str,
    sweep_mode: str,
    top_k: int,
    ef_list: Iterable[Optional[int]],
    beam_list: Iterable[Optional[int]],
    pending_list: Iterable[Optional[int]],
) -> Iterator[ParamPoint]:
    if sweep_mode == "off":
        ef = next(iter(ef_list), None)
        beam_width = next(iter(beam_list), None)
        max_pending_bytes = next(iter(pending_list), None)
        yield ParamPoint(top_k=top_k, ef=ef, beam_width=beam_width, max_pending_bytes=max_pending_bytes)
        return

    # `recommended` is a curated subset (only top_k=10) for fast regression
    # snapshots; `spec` is the full matrix declared in proposal.md (top_k
    # ∈ {1, 10, 100}). Both share the per-engine axes (ef / beam_width /
    # max_pending_bytes).
    if sweep_mode == "recommended":
        top_ks: tuple[int, ...] = (10,)
    elif sweep_mode == "spec":
        top_ks = SWEEP_TOP_K
    else:
        raise ValueError(f"unknown sweep mode: {sweep_mode}")

    if engine == "disk_flat":
        for k in top_ks:
            for pending in SWEEP_PENDING_BYTES:
                yield ParamPoint(top_k=k, ef=None, beam_width=None, max_pending_bytes=pending)
    elif engine == "disk_vamana":
        for k in top_ks:
            for ef in SWEEP_EF:
                yield ParamPoint(top_k=k, ef=ef, beam_width=None, max_pending_bytes=None)
    elif engine == "disk_laser":
        for k in top_ks:
            for ef in SWEEP_EF:
                for beam_width in SWEEP_BEAM_WIDTH:
                    yield ParamPoint(top_k=k, ef=ef, beam_width=beam_width, max_pending_bytes=None)
    else:
        raise ValueError(f"unknown engine: {engine}")


def _filter_ignored_args(engine: str, params: dict) -> list[str]:
    ignored = []
    metric = params.get("metric")
    vamana_only = ("vamana_R", "vamana_L", "vamana_alpha", "vamana_seed")
    if engine == "disk_flat":
        if params.get("ef") is not None:
            ignored.append("ef")
        if params.get("beam_width") is not None:
            ignored.append("beam_width")
        ignored.extend(name for name in vamana_only if params.get(name) is not None)
    elif engine == "disk_vamana":
        if params.get("beam_width") is not None:
            ignored.append("beam_width")
        if params.get("max_pending_bytes") is not None:
            ignored.append("max_pending_bytes")
        if metric in {"IP", "COS"}:
            ignored.append(f"metric={metric}")
    elif engine == "disk_laser":
        if params.get("max_pending_bytes") is not None:
            ignored.append("max_pending_bytes")
        if metric in {"IP", "COS"}:
            ignored.append(f"metric={metric}")
        ignored.extend(name for name in vamana_only if params.get(name) is not None)
    else:
        raise ValueError(f"unknown engine: {engine}")
    return ignored


def _effective_params(engine: str, params: dict) -> tuple[str, dict, dict, list[str]]:
    ignored = _filter_ignored_args(engine, params)
    metric = params["metric"] if engine == "disk_flat" else "L2"
    engine_params = {
        **params,
        "metric": metric,
        "ef": params.get("ef") if engine in {"disk_vamana", "disk_laser"} else None,
        "beam_width": params.get("beam_width") if engine == "disk_laser" else None,
        "max_pending_bytes": params.get("max_pending_bytes") if engine == "disk_flat" else None,
    }
    raw_params: dict = {
        "top_k": params["top_k"],
        "ef": engine_params["ef"],
        "beam_width": engine_params["beam_width"],
        "max_pending_bytes": (
            ("default" if engine_params["max_pending_bytes"] is None else engine_params["max_pending_bytes"])
            if engine == "disk_flat"
            else None
        ),
    }
    if engine == "disk_vamana":
        raw_params["vamana_R"] = int(params["vamana_R"])
        raw_params["vamana_L"] = int(params["vamana_L"])
        raw_params["vamana_alpha"] = float(params["vamana_alpha"])
        raw_params["vamana_seed"] = int(params["vamana_seed"])
    return metric, engine_params, raw_params, ignored


def _default_run_id() -> str:
    # Microsecond precision so back-to-back invocations within the same
    # second still get distinct ids; collision-handling in _write_outputs
    # is the second line of defense (e.g. when --run-id is user-supplied
    # and reused intentionally or by mistake).
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    rev = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    sha = rev.stdout.strip()[:8] if rev.returncode == 0 else "unknown"
    return f"{stamp}_{sha}"


def _allocate_run_dir(out_root: Path, run_id: str) -> Path:
    """Return `out_root/run_id` if free, else `out_root/run_id_NNN` (3-digit)."""
    candidate = Path(out_root) / run_id
    if not (candidate / "summary.json").exists():
        return candidate
    for i in range(1, 1000):
        suffixed = Path(out_root) / f"{run_id}_{i:03d}"
        if not (suffixed / "summary.json").exists():
            return suffixed
    raise RuntimeError(f"could not find a free run dir for {run_id} after 999 attempts")


def _load_dataset(name: str, args: argparse.Namespace, metric: str) -> DatasetSpec:
    if name == "laser_files":
        return load_laser_files(
            n=args.n,
            dim=args.dim,
            query_count=args.queries,
            seed=args.seed,
            vectors_path=args.vectors,
            queries_path=args.queries_path,
            ground_truth_path=args.ground_truth,
            laser_src_dir=args.laser_src_dir,
        )
    if name == "synth":
        return load_synth(args.n, args.dim, args.queries, args.seed, metric=metric)

    root = args.dataset_root or os.environ.get("ALAYALITE_BENCH_DATASET_ROOT")
    if not root:
        raise DatasetMissing(f"{name} requires --dataset-root or ALAYALITE_BENCH_DATASET_ROOT")
    if name == "sift1m":
        return load_sift1m(Path(root))
    if name == "gist1m":
        return load_gist1m(Path(root))
    raise ValueError(f"unknown dataset: {name}")


def _raw_filename(raw: dict, sweep_mode: str) -> str:
    base = f"{raw['engine']}_{raw['dataset']}_{raw['metric']}"
    if raw.get("status") == "skipped" and raw["engine"] == "disk_laser":
        # Include dataset + metric so multi-dataset runs ("--dataset synth
        # --dataset sift1m") on a LASER-disabled build emit one skip raw per
        # (dataset, metric) instead of overwriting a shared file.
        return f"{base}_skip.json"

    if sweep_mode == "off":
        return f"{base}.json"

    params = raw["params"]
    pieces = [f"k{params['top_k']}"]
    if params.get("ef") is not None:
        pieces.append(f"ef{params['ef']}")
    if params.get("beam_width") is not None:
        pieces.append(f"bw{params['beam_width']}")
    if raw["engine"] == "disk_flat":
        pieces.append(f"pending{params['max_pending_bytes']}")
    if raw["engine"] == "disk_vamana":
        pieces.append(f"R{params['vamana_R']}_L{params['vamana_L']}")
    return f"{base}_{'_'.join(pieces)}.json"


def _write_outputs(
    out_root: Path,
    run_id: str,
    summary_json: dict,
    summary_md: str,
    raws: list[dict],
    *,
    sweep_mode: str,
) -> Path:
    run_dir = _allocate_run_dir(out_root, run_id)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for raw in raws:
        write_json(raw_dir / _raw_filename(raw, sweep_mode), raw)
    write_json(run_dir / "summary.json", summary_json)
    (run_dir / "summary.md").write_text(summary_md + "\n", encoding="utf-8")
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", action="append", choices=ENGINES, default=None)
    parser.add_argument("--dataset", action="append", choices=DATASETS, default=None)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", choices=METRICS, default="L2")
    parser.add_argument("--ef", type=int, default=100)
    parser.add_argument("--beam-width", dest="beam_width", type=int, default=4)
    parser.add_argument("--max-pending-bytes", type=_parse_max_pending_bytes, default=None)
    # Vamana build parameters. Defaults match the deleted disk_vamana_smoke.py
    # (R=64, L=200, alpha=1.2, seed=42) so historical Vamana numbers are
    # cross-comparable.
    parser.add_argument(
        "--vamana-R", dest="vamana_R", type=int, default=64, help="disk_vamana: per-vertex neighbour cap (default 64)."
    )
    parser.add_argument(
        "--vamana-L", dest="vamana_L", type=int, default=100, help="disk_vamana: build search list width (default 100)."
    )
    parser.add_argument(
        "--vamana-alpha",
        dest="vamana_alpha",
        type=float,
        default=1.2,
        help="disk_vamana: pruning slack alpha (default 1.2).",
    )
    parser.add_argument(
        "--vamana-seed",
        dest="vamana_seed",
        type=int,
        default=42,
        help="disk_vamana: deterministic build seed (default 42).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--laser-src-dir", type=Path, default=None)
    parser.add_argument("--vectors", type=Path, default=None, help="LASER legacy input vectors (.fbin).")
    parser.add_argument("--queries-path", type=Path, default=None, help="LASER legacy query vectors (.fbin).")
    parser.add_argument("--ground-truth", type=Path, default=None, help="LASER legacy ground-truth labels (.ibin).")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--sweep", choices=("off", "recommended", "spec"), default="off")
    return parser


def _skip_result(reason: str) -> dict:
    return {
        "recall_at_1": None,
        "recall_at_10": None,
        "recall_at_100": None,
        "recall_status": "skipped",
        "qps": None,
        "latency_us": {"p50": None, "p95": None, "p99": None, "min": None, "mean": None},
        "build_wall_s": None,
        "on_disk_bytes": None,
        "peak_rss_kb": None,
        "peak_rss_unit": None,
        "segment_count": None,
        "reason": reason,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    engines = args.engine or list(ENGINES)
    datasets = args.dataset or ["synth"]
    run_id = args.run_id or _default_run_id()
    harness_argv = [sys.argv[0], *(list(argv) if argv is not None else sys.argv[1:])]
    run_dir = args.out / run_id
    scratch_root = run_dir / "_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    raws: list[dict] = []
    laser_supported: Optional[bool] = None

    for engine in engines:
        for dataset_name in datasets:
            raw_metric = args.metric
            effective_metric = raw_metric if engine == "disk_flat" else "L2"
            try:
                dataset = _load_dataset(dataset_name, args, effective_metric)
            except DatasetMissing:
                print(f"engine={engine} dataset={dataset_name} status=skipped reason=missing_dataset")
                continue

            provenance = collect_provenance(args.seed, dataset.sha16, harness_argv)

            if engine == "disk_laser":
                if laser_supported is None:
                    laser_supported = probe_disk_laser_supported(str(scratch_root))
                if not laser_supported:
                    print("engine=disk_laser status=skipped reason=probe_failed")
                    raw = render_raw_json(
                        _skip_result("probe_failed"),
                        provenance,
                        {"top_k": args.k, "ef": None, "beam_width": None, "max_pending_bytes": None},
                        engine,
                        dataset_name,
                        effective_metric,
                        dataset.n,
                        dataset.dim,
                        args.queries,
                        run_id=run_id,
                        ignored_args=[],
                        status="skipped",
                        reason="probe_failed",
                    )
                    raws.append(raw)
                    continue

            points = _iter_sweep_points(
                engine,
                args.sweep,
                args.k,
                [args.ef],
                [args.beam_width],
                [args.max_pending_bytes],
            )
            for point in points:
                cli_params = {
                    "top_k": point.top_k,
                    "ef": point.ef,
                    "beam_width": point.beam_width,
                    "max_pending_bytes": point.max_pending_bytes,
                    "metric": raw_metric,
                    "queries": args.queries,
                    "warmup": args.warmup,
                    "laser_src_dir": str(args.laser_src_dir) if args.laser_src_dir else None,
                    "laser_recall_valid": engine == "disk_laser"
                    and dataset_name == "laser_files"
                    and dataset.ground_truth is not None,
                    "scratch_root": str(scratch_root),
                    "vamana_R": args.vamana_R,
                    "vamana_L": args.vamana_L,
                    "vamana_alpha": args.vamana_alpha,
                    "vamana_seed": args.vamana_seed,
                }
                metric, engine_params, raw_params, ignored_args = _effective_params(engine, cli_params)
                if engine == "disk_flat":
                    result = bench_disk_flat(dataset, engine_params)
                elif engine == "disk_vamana":
                    result = bench_disk_vamana(dataset, engine_params)
                elif engine == "disk_laser":
                    result = bench_disk_laser(dataset, engine_params)
                else:
                    raise ValueError(f"unknown engine: {engine}")

                raws.append(
                    render_raw_json(
                        result,
                        provenance,
                        raw_params,
                        engine,
                        dataset_name,
                        metric,
                        dataset.n,
                        dataset.dim,
                        args.queries,
                        run_id=run_id,
                        ignored_args=ignored_args,
                    )
                )

    summary_provenance = (
        raws[0]["provenance"] if raws else collect_provenance(args.seed, "0000000000000000", harness_argv)
    )
    summary = render_summary_json(raws, run_id, summary_provenance)
    shutil.rmtree(scratch_root, ignore_errors=True)
    _write_outputs(args.out, run_id, summary, render_summary_md(summary), raws, sweep_mode=args.sweep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
