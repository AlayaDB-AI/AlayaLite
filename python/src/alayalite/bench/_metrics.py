# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metric and report rendering helpers for DiskCollection benchmarks."""

# pylint: disable=inconsistent-quotes  # Python 3.10 forbids same-quote nesting
#                                        in f-strings; subscript access in
#                                        f-strings keeps single quotes inside.
# pylint: disable=too-many-positional-arguments  # render_raw_json wraps a
#                                                  flat schema; design D4.

from __future__ import annotations

import json
import resource
import statistics
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

SCHEMA_VERSION = 1


def percentiles(values: Iterable[float], qs: tuple[int, ...] = (50, 95, 99)) -> dict[int, float]:
    sorted_values = sorted(float(v) for v in values)
    if not sorted_values:
        raise ValueError("percentiles requires at least one value")
    out: dict[int, float] = {}
    for q in qs:
        idx = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * q / 100)))
        out[q] = sorted_values[idx]
    return out


def recall_at_k(predicted_ids: Iterable[Iterable[int]], ground_truth_ids: np.ndarray, k: int) -> float:
    # Materialise predictions so we can length-check before zip; otherwise
    # mismatched lengths silently drop queries from the recall mean.
    predicted_list = [list(p) for p in predicted_ids]
    if len(predicted_list) != len(ground_truth_ids):
        raise ValueError(
            f"recall_at_k: predictions ({len(predicted_list)}) and ground truth "
            f"({len(ground_truth_ids)}) row counts must match"
        )
    if not predicted_list:
        raise ValueError("recall_at_k requires at least one query")
    recalls = []
    for predicted, ground_truth in zip(predicted_list, ground_truth_ids):
        pred_topk = {int(v) for v in predicted[:k]}
        gt_topk = {int(v) for v in list(ground_truth)[:k]}
        recalls.append(len(pred_topk & gt_topk) / float(k))
    return float(statistics.mean(recalls))


def segment_bytes(col_path: Path) -> int:
    segments = Path(col_path) / "segments"
    if not segments.exists():
        return 0
    return sum(path.stat().st_size for path in segments.rglob("*") if path.is_file())


def segment_count(col_path: Path) -> int:
    segments = Path(col_path) / "segments"
    if not segments.exists():
        return 0
    return sum(1 for path in segments.iterdir() if path.is_dir() and path.name.startswith("seg_"))


def peak_rss_kb_and_unit() -> tuple[int, str]:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return int((rss + 1023) // 1024), "bytes_converted_to_kb"
    return rss, "KB"


def latency_summary_us(latencies_us: list[float]) -> dict[str, float]:
    pct = percentiles(latencies_us)
    return {
        "p50": pct[50],
        "p95": pct[95],
        "p99": pct[99],
        "min": float(min(latencies_us)),
        "mean": float(statistics.mean(latencies_us)),
    }


def render_raw_json(
    result: dict,
    provenance: dict,
    params: dict,
    engine: str,
    dataset: str,
    metric: str,
    n: int,
    dim: int,
    queries: int,
    *,
    run_id: str,
    ignored_args: Optional[list[str]] = None,
    status: str = "ok",
    reason: Optional[str] = None,
) -> dict:
    raw = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "engine": engine,
        "dataset": dataset,
        "metric": metric,
        "params": params,
        "n": int(n),
        "dim": int(dim),
        "queries": int(queries),
        "ignored_args": list(ignored_args or []),
        "results": result,
        "provenance": provenance,
    }
    if status != "ok":
        raw["status"] = status
    if reason is not None:
        raw["reason"] = reason
    return raw


def _summary_provenance(raws: list[dict], provenance: dict) -> dict:
    out = deepcopy(provenance)
    hashes = sorted({raw["provenance"]["dataset_sha256_prefix"] for raw in raws if raw.get("provenance")})
    if not hashes:
        out["dataset_sha256_prefix"] = "0000000000000000"
        out["dataset_sha256_prefix_source"] = "no_dataset"
        out["dataset_sha256_prefix_members"] = []
    elif len(hashes) == 1:
        out["dataset_sha256_prefix"] = hashes[0]
        out["dataset_sha256_prefix_source"] = "single_dataset"
        out["dataset_sha256_prefix_members"] = hashes
    else:
        out["dataset_sha256_prefix"] = sha256("".join(hashes).encode("ascii")).hexdigest()[:16]
        out["dataset_sha256_prefix_source"] = "aggregate_multiple_datasets"
        out["dataset_sha256_prefix_members"] = hashes
    return out


def render_summary_json(raws: list[dict], run_id: str, provenance: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "provenance": _summary_provenance(raws, provenance),
        "raws": raws,
    }


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_summary_md(summary_json: dict) -> str:
    provenance = summary_json["provenance"]
    lines = [
        "# DiskCollection Benchmark Summary",
        "",
        f"- schema_version: {summary_json['schema_version']}",
        f"- run_id: {summary_json['run_id']}",
        f"- git_commit_sha: {provenance['git_commit_sha']}",
        f"- git_dirty: {provenance['git_dirty']}",
        f"- timestamp_iso8601: {provenance['timestamp_iso8601']}",
        f"- cpu_model: {provenance['cpu_model']}",
        f"- cpu_count: {provenance['cpu_count']}",
        f"- mem_total_kb: {provenance['mem_total_kb']}",
        f"- compiler_flags: {provenance['compiler_flags']}",
        "",
    ]

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for raw in summary_json["raws"]:
        grouped.setdefault((raw["engine"], raw["dataset"], raw.get("metric", "L2")), []).append(raw)

    for (engine, dataset, metric), raws in sorted(grouped.items()):
        lines.extend(
            [
                f"## {engine} / {dataset} / {metric}",
                "",
                (
                    "| status | k | ef | beam_width | max_pending_bytes | recall@10"
                    " | qps | p50_us | p99_us | build_s | bytes | segments |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for raw in raws:
            params = raw.get("params", {})
            results = raw.get("results", {})
            latency = results.get("latency_us") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        _fmt(raw.get("status", "ok")),
                        _fmt(params.get("top_k")),
                        _fmt(params.get("ef")),
                        _fmt(params.get("beam_width")),
                        _fmt(params.get("max_pending_bytes")),
                        _fmt(results.get("recall_at_10")),
                        _fmt(results.get("qps")),
                        _fmt(latency.get("p50")),
                        _fmt(latency.get("p99")),
                        _fmt(results.get("build_wall_s")),
                        _fmt(results.get("on_disk_bytes")),
                        _fmt(results.get("segment_count")),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
