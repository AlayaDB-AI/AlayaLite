#!/usr/bin/env python3
"""
Benchmark ``Collection.hybrid_query`` on a real hybrid-search dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
from alayalite import Collection
from alayalite.schema import IndexParams
from alayalite.utils import load_npy_jsonl_dataset


def load_query_vectors(tests_path: Path, query_num: int) -> np.ndarray:
    queries = []
    with tests_path.open(encoding="utf-8") as tests_file:
        for line in tests_file:
            if len(queries) >= query_num:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            vector = payload.get("vector") or payload.get("query") or payload.get("query_vector")
            if vector is None:
                raise ValueError("tests.jsonl row is missing a query vector")
            queries.append(vector)

    if len(queries) != query_num:
        raise ValueError(f"Requested {query_num} queries, but only found {len(queries)}")

    return np.asarray(queries, dtype=np.float32)


def detect_numeric_field(payloads_path: Path) -> str:
    with payloads_path.open(encoding="utf-8") as payloads_file:
        for line in payloads_file:
            if not line.strip():
                continue
            payload = json.loads(line)
            for key, value in payload.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    return key
    raise ValueError("No numeric payload field found for selectivity filter")


def load_numeric_values(payloads_path: Path, field: str) -> np.ndarray:
    values = []
    with payloads_path.open(encoding="utf-8") as payloads_file:
        for line in payloads_file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if field not in payload:
                raise ValueError(f"Missing numeric field '{field}' in payload row")
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Field '{field}' is not consistently numeric")
            values.append(value)
    return np.asarray(values)


def choose_threshold(values: np.ndarray, selectivity: float):
    if not 0 < selectivity <= 1:
        raise ValueError("selectivity must be in (0, 1]")
    rank = max(0, int(np.ceil(values.shape[0] * selectivity)) - 1)
    threshold = np.partition(values.copy(), rank)[rank]
    actual = float(np.mean(values <= threshold))
    if np.issubdtype(values.dtype, np.integer):
        return int(threshold), actual
    return float(threshold), actual


def make_collection(
    name: str,
    rocksdb_path: str,
    data_num: int,
    filter_field: str,
    build_threads: int,
) -> Collection:
    params = IndexParams(
        index_type="hnsw",
        quantization_type="none",
        metric="cos",
        capacity=int(data_num),
        max_nbrs=32,
        build_threads=build_threads,
        rocksdb_path=rocksdb_path,
        indexed_fields=[filter_field],
    )
    return Collection(name, params)


def benchmark(args: argparse.Namespace) -> dict:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    vectors_path = dataset_dir / "vectors.npy"
    payloads_path = dataset_dir / "payloads.jsonl"
    tests_path = dataset_dir / "tests.jsonl"
    for path in (vectors_path, payloads_path, tests_path):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

    filter_field = args.filter_field or detect_numeric_field(payloads_path)
    numeric_values = load_numeric_values(payloads_path, filter_field)
    threshold, actual_selectivity = choose_threshold(numeric_values, args.selectivity)
    del numeric_values

    dataset = load_npy_jsonl_dataset(
        str(vectors_path),
        str(payloads_path),
        item_id_field=args.item_id_field,
        document_field=args.document_field,
        metadata_fields=[filter_field],
    )
    queries = load_query_vectors(tests_path, args.query_num)
    data_num = int(dataset.vectors.shape[0])
    dim = int(dataset.vectors.shape[1])

    filter_dict = {filter_field: {"$le": threshold}}
    round_results = []

    with tempfile.TemporaryDirectory(prefix="alayalite-raw-hybrid-bench-") as temp_dir:
        rocksdb_path = os.path.join(temp_dir, "rocksdb")
        collection = make_collection(
            name="raw_hybrid_collection_perf",
            rocksdb_path=rocksdb_path,
            data_num=data_num,
            filter_field=filter_field,
            build_threads=args.build_threads,
        )
        try:
            build_start = time.perf_counter()
            collection.fit(
                dataset.vectors,
                item_ids=dataset.item_ids,
                documents=dataset.documents,
                metadata_list=dataset.metadata_list,
                num_threads=args.build_threads,
            )
            build_seconds = time.perf_counter() - build_start
            compiled_filter = collection.build_filter(filter_dict)

            del dataset

            collection.hybrid_query(
                queries,
                limit=args.topk,
                metadata_filter=compiled_filter,
                ef_search=args.ef_search,
                num_threads=args.query_threads,
                filter_execution_hint="auto",
            )

            for round_id in range(args.rounds):
                start = time.perf_counter()
                result = collection.hybrid_query(
                    queries,
                    limit=args.topk,
                    metadata_filter=compiled_filter,
                    ef_search=args.ef_search,
                    num_threads=args.query_threads,
                    filter_execution_hint="auto",
                )
                elapsed = time.perf_counter() - start
                rows = result["id"]
                if len(rows) != args.query_num:
                    raise RuntimeError(f"Expected {args.query_num} query rows, got {len(rows)}")
                round_results.append(
                    {
                        "round": round_id + 1,
                        "query_seconds": elapsed,
                        "avg_ms_per_query": elapsed * 1000.0 / args.query_num,
                        "qps": args.query_num / elapsed if elapsed > 0 else float("inf"),
                    }
                )
        finally:
            collection.close()

    median_query_seconds = statistics.median(entry["query_seconds"] for entry in round_results)
    median_avg_ms = statistics.median(entry["avg_ms_per_query"] for entry in round_results)
    median_qps = statistics.median(entry["qps"] for entry in round_results)

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "dataset": {
            "name": dataset_dir.name,
            "path": str(dataset_dir),
            "data_num": data_num,
            "dim": dim,
            "query_source": "tests.jsonl",
        },
        "benchmark": {
            "query_num": args.query_num,
            "topk": args.topk,
            "ef_search": args.ef_search,
            "rounds": args.rounds,
            "build_threads": args.build_threads,
            "query_threads": args.query_threads,
            "build_seconds": build_seconds,
            "median_query_seconds": median_query_seconds,
            "median_avg_ms_per_query": median_avg_ms,
            "median_qps": median_qps,
            "round_results": round_results,
        },
        "filter": {
            "field": filter_field,
            "operator": "$le",
            "threshold": threshold,
            "target_selectivity": args.selectivity,
            "actual_selectivity": actual_selectivity,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Collection.hybrid_query on a real dataset")
    parser.add_argument(
        "--dataset-dir", required=True, help="Directory containing vectors.npy/payloads.jsonl/tests.jsonl"
    )
    parser.add_argument("--query-num", type=int, default=100)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--selectivity", type=float, default=0.01)
    parser.add_argument("--build-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--query-threads", type=int, default=1)
    parser.add_argument("--filter-field", default=None)
    parser.add_argument("--item-id-field", default=None)
    parser.add_argument("--document-field", default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = benchmark(args)

    output_text = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")
    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
