# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Per-engine benchmark adapters for `alayalite.DiskCollection`."""

# pylint: disable=invalid-name  # underscore-prefixed unused vars
#                                 (`_params`, `_distance`) follow Python's
#                                 unused-variable convention but pylint's
#                                 default name regex `^[a-z]...` rejects
#                                 the leading underscore.

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from alayalite import DiskCollection, MetricType

from ._datasets import DatasetSpec
from ._metrics import latency_summary_us, peak_rss_kb_and_unit, recall_at_k, segment_bytes, segment_count

DEFAULT_MAX_PENDING_BYTES = 512 * 1024 * 1024


def _temp_dir(prefix: str, scratch_root: Optional[str] = None):
    if scratch_root:
        Path(scratch_root).mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=scratch_root)


def probe_disk_laser_supported(scratch_root: Optional[str] = None) -> bool:
    """Attempt a tiny disk_laser constructor; return True iff it succeeds."""
    if scratch_root:
        Path(scratch_root).mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="alayalite_laser_probe_", dir=scratch_root))
    target = tmp / "probe"
    try:
        DiskCollection(
            path=str(target),
            dim=128,
            metric=MetricType.L2,
            index_type="disk_laser",
        )
    except (ValueError, RuntimeError) as exc:
        msg = str(exc)
        if "disk_laser" in msg and "not implemented in v1" in msg:
            return False
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return True


def _metric_type(metric: str):
    return {"L2": MetricType.L2, "IP": MetricType.IP, "COS": MetricType.COS}[metric]


def _batch_rows(dim: int, max_pending_bytes: Optional[int], n: int) -> int:
    # DiskCollection.add() (include/index/disk/disk_collection.hpp:374-393)
    # rejects any single batch where 2 * n * per_row > max_pending_bytes
    # ("defense in depth: 2x for swap-window memory peak"). The harness must
    # mirror that 2x to keep the disk_flat fan-out sweep (small
    # --max-pending-bytes) from crashing in add().
    cap = DEFAULT_MAX_PENDING_BYTES if max_pending_bytes is None else int(max_pending_bytes)
    row_bytes = dim * np.dtype(np.float32).itemsize + np.dtype(np.uint64).itemsize
    return max(1, min(n, cap // (row_bytes * 2)))


def _build_disk_flat(
    col_path: Path, dataset: DatasetSpec, metric: str, max_pending_bytes: Optional[int]
) -> tuple[object, float]:
    kwargs = {}
    if max_pending_bytes is not None:
        kwargs["max_pending_bytes"] = int(max_pending_bytes)
    col = DiskCollection(
        path=str(col_path),
        dim=dataset.dim,
        metric=_metric_type(metric),
        index_type="disk_flat",
        **kwargs,
    )

    rows_per_batch = _batch_rows(dataset.dim, max_pending_bytes, dataset.n)
    start = time.perf_counter()
    for offset in range(0, dataset.n, rows_per_batch):
        end = min(offset + rows_per_batch, dataset.n)
        col.add(
            np.ascontiguousarray(dataset.vectors[offset:end], dtype=np.float32),
            np.ascontiguousarray(dataset.labels[offset:end], dtype=np.uint64),
        )
        col.flush()
    return col, time.perf_counter() - start


def _build_disk_vamana(col_path: Path, dataset: DatasetSpec, params: dict) -> tuple[object, float]:
    col = DiskCollection(
        path=str(col_path),
        dim=dataset.dim,
        metric=MetricType.L2,
        index_type="disk_vamana",
        vamana_R=int(params["vamana_R"]),
        vamana_L=int(params["vamana_L"]),
        vamana_alpha=float(params["vamana_alpha"]),
        vamana_seed=int(params["vamana_seed"]),
        vamana_num_threads=1,
    )
    start = time.perf_counter()
    col.add(
        np.ascontiguousarray(dataset.vectors, dtype=np.float32), np.ascontiguousarray(dataset.labels, dtype=np.uint64)
    )
    col.flush()
    return col, time.perf_counter() - start


def _build_disk_laser(col_path: Path, dataset: DatasetSpec, src_dir: Path) -> tuple[object, float]:
    col = DiskCollection(
        path=str(col_path),
        dim=dataset.dim,
        metric=MetricType.L2,
        index_type="disk_laser",
    )
    start = time.perf_counter()
    col.import_laser_segment(str(src_dir), np.ascontiguousarray(dataset.labels, dtype=np.uint64))
    return col, time.perf_counter() - start


def _search_flat(col, query: np.ndarray, k: int, _params: dict):
    return col.search(query, k=k)


def _search_vamana(col, query: np.ndarray, k: int, params: dict):
    return col.search(query, k=k, ef=int(params["ef"]))


def _search_laser(col, query: np.ndarray, k: int, params: dict):
    return col.search(query, k=k, ef=int(params["ef"]), beam_width=int(params["beam_width"]))


def _measure_search(
    col,
    dataset: DatasetSpec,
    params: dict,
    search: Callable[[object, np.ndarray, int, dict], list],
) -> tuple[dict, list[list[int]]]:
    timed_queries = int(params["queries"])
    warmup = int(params["warmup"])
    top_k = int(params["top_k"])
    if dataset.queries.shape[0] < timed_queries:
        raise ValueError(f"dataset has {dataset.queries.shape[0]} queries, fewer than requested {timed_queries}")

    for i in range(warmup):
        query = np.ascontiguousarray(dataset.queries[i % timed_queries], dtype=np.float32)
        search(col, query, top_k, params)

    latencies_us: list[float] = []
    timed_predictions: list[list[int]] = []
    start_loop = time.perf_counter()
    for i in range(timed_queries):
        query = np.ascontiguousarray(dataset.queries[i], dtype=np.float32)
        start = time.perf_counter()
        hits = search(col, query, top_k, params)
        latencies_us.append((time.perf_counter() - start) * 1e6)
        timed_predictions.append([int(label) for label, _distance in hits])
    elapsed = time.perf_counter() - start_loop

    recall_depth = min(100, dataset.n)
    recall_predictions = timed_predictions
    if top_k < recall_depth:
        recall_predictions = []
        for i in range(timed_queries):
            query = np.ascontiguousarray(dataset.queries[i], dtype=np.float32)
            hits = search(col, query, recall_depth, params)
            recall_predictions.append([int(label) for label, _ in hits])

    results = {
        "qps": float(timed_queries / elapsed),
        "latency_us": latency_summary_us(latencies_us),
    }
    return results, recall_predictions


def _finish_results(
    base: dict,
    dataset: DatasetSpec,
    predictions: list[list[int]],
    build_wall_s: float,
    col_path: Path,
    *,
    allow_recall: bool = True,
) -> dict:
    if not allow_recall:
        recall = {
            "recall_at_1": None,
            "recall_at_10": None,
            "recall_at_100": None,
            "recall_status": "skipped",
        }
    elif dataset.ground_truth is None:
        recall = {
            "recall_at_1": None,
            "recall_at_10": None,
            "recall_at_100": None,
            "recall_status": "missing_ground_truth",
        }
    else:
        gt = dataset.ground_truth[: len(predictions)]
        recall = {
            "recall_at_1": recall_at_k(predictions, gt, 1),
            "recall_at_10": recall_at_k(predictions, gt, 10),
            "recall_at_100": recall_at_k(predictions, gt, 100),
            "recall_status": "computed",
        }
    rss_kb, rss_unit = peak_rss_kb_and_unit()
    return {
        **recall,
        **base,
        "build_wall_s": float(build_wall_s),
        "on_disk_bytes": int(segment_bytes(col_path)),
        "peak_rss_kb": int(rss_kb),
        "peak_rss_unit": rss_unit,
        "segment_count": int(segment_count(col_path)),
    }


def bench_disk_flat(dataset: DatasetSpec, params: dict) -> dict:
    with _temp_dir("alayalite_bench_disk_flat_", params.get("scratch_root")) as tmp:
        col_path = Path(tmp) / "collection"
        col, build_wall_s = _build_disk_flat(col_path, dataset, params["metric"], params.get("max_pending_bytes"))
        base, predictions = _measure_search(col, dataset, params, _search_flat)
        return _finish_results(base, dataset, predictions, build_wall_s, col_path)


def bench_disk_vamana(dataset: DatasetSpec, params: dict) -> dict:
    with _temp_dir("alayalite_bench_disk_vamana_", params.get("scratch_root")) as tmp:
        col_path = Path(tmp) / "collection"
        col, build_wall_s = _build_disk_vamana(col_path, dataset, params)
        base, predictions = _measure_search(col, dataset, params, _search_vamana)
        return _finish_results(base, dataset, predictions, build_wall_s, col_path)


def bench_disk_laser(dataset: DatasetSpec, params: dict) -> dict:
    src_dir = params.get("laser_src_dir")
    if not src_dir:
        raise ValueError("--laser-src-dir is required when disk_laser is supported and requested")

    with _temp_dir("alayalite_bench_disk_laser_", params.get("scratch_root")) as tmp:
        col_path = Path(tmp) / "collection"
        col, build_wall_s = _build_disk_laser(col_path, dataset, Path(src_dir))
        base, predictions = _measure_search(col, dataset, params, _search_laser)
        return _finish_results(
            base,
            dataset,
            predictions,
            build_wall_s,
            col_path,
            allow_recall=bool(params.get("laser_recall_valid")),
        )
