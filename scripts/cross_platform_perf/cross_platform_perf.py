#!/usr/bin/env python3
"""CI helper for the cross-platform performance workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET = "qdrant_random_ints_1m"
DEFAULT_DATASET_DIR_NAME = "qdrant-random-ints-1m"
DEFAULT_LABELS = [
    "ubuntu-latest",
    "ubuntu-24.04-arm",
    "macos-15-intel",
    "macos-14",
    "windows-2022",
]


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _summary_path() -> Path:
    try:
        return Path(os.environ["GITHUB_STEP_SUMMARY"])
    except KeyError as exc:
        raise SystemExit("GITHUB_STEP_SUMMARY is not set") from exc


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def _artifact_path(label: str) -> Path:
    return Path("artifacts") / f"raw-hybrid-collection-{label}.json"


def download_dataset(args: argparse.Namespace) -> int:
    return _run(
        [
            "uv",
            "run",
            "python",
            "scripts/benchmarks/download_qdrant_filtered_dataset.py",
            "--dataset",
            args.dataset,
            "--output-dir",
            str(args.output_dir),
        ]
    )


def run_benchmark(args: argparse.Namespace) -> int:
    output_path = args.output or _artifact_path(args.label)
    dataset_dir = args.dataset_dir or (args.dataset_root / DEFAULT_DATASET_DIR_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/benchmarks/raw_hybrid_collection_benchmark.py",
        "--dataset-dir",
        str(dataset_dir),
        "--query-num",
        args.query_num,
        "--topk",
        args.topk,
        "--ef-search",
        args.ef_search,
        "--rounds",
        args.rounds,
        "--selectivity",
        args.selectivity,
        "--query-threads",
        args.query_threads,
        "--output",
        str(output_path),
    ]
    if args.build_threads != "0":
        cmd.extend(["--build-threads", args.build_threads])

    return _run(cmd)


def _single_summary_lines(result: dict) -> list[str]:
    benchmark = result["benchmark"]
    dataset = result["dataset"]
    filter_info = result["filter"]
    platform_info = result["platform"]

    lines = [
        f"## Raw Hybrid Collection Benchmark ({platform_info['system']} / {platform_info['machine']})",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Dataset | `{dataset['name']}` |",
        f"| Data size | `{dataset['data_num']}` |",
        f"| Dimension | `{dataset['dim']}` |",
        f"| Queries | `{benchmark['query_num']}` |",
        f"| Top-k | `{benchmark['topk']}` |",
        f"| ef_search | `{benchmark['ef_search']}` |",
        f"| Rounds | `{benchmark['rounds']}` |",
        f"| Filter | `{filter_info['field']} <= {filter_info['threshold']}` |",
        f"| Target selectivity | `{filter_info['target_selectivity']:.4f}` |",
        f"| Actual selectivity | `{filter_info['actual_selectivity']:.4f}` |",
        f"| Build seconds | `{benchmark['build_seconds']:.3f}` |",
        f"| Median query seconds | `{benchmark['median_query_seconds']:.3f}` |",
        f"| Median ms/query | `{benchmark['median_avg_ms_per_query']:.3f}` |",
        f"| Median QPS | `{benchmark['median_qps']:.3f}` |",
        "",
        "### Per-round results",
        "",
        "| Round | Query Seconds | Avg ms/query | QPS |",
        "| --- | --- | --- | --- |",
    ]

    for round_result in benchmark["round_results"]:
        lines.append(
            f"| {round_result['round']} | "
            f"{round_result['query_seconds']:.3f} | "
            f"{round_result['avg_ms_per_query']:.3f} | "
            f"{round_result['qps']:.3f} |"
        )
    return lines


def append_summary(args: argparse.Namespace) -> int:
    result_path = args.result or _artifact_path(args.label)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with _summary_path().open("a", encoding="utf-8") as summary:
        summary.write("\n".join(_single_summary_lines(result)))
        summary.write("\n")
    return 0


def _aggregate_summary_lines(result_paths: list[Path], expected_labels: list[str]) -> list[str]:
    if not result_paths:
        raise SystemExit("No benchmark artifacts found to aggregate.")

    label_order = {label: index for index, label in enumerate(expected_labels)}
    results = []
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        benchmark = result["benchmark"]
        dataset = result["dataset"]
        filter_info = result["filter"]
        platform_info = result["platform"]
        label = result_path.stem.removeprefix("raw-hybrid-collection-")
        results.append(
            {
                "label": label,
                "platform": f"{platform_info['system']} / {platform_info['machine']}",
                "dataset_name": dataset["name"],
                "data_num": dataset["data_num"],
                "dim": dataset["dim"],
                "query_num": benchmark["query_num"],
                "topk": benchmark["topk"],
                "ef_search": benchmark["ef_search"],
                "rounds": benchmark["rounds"],
                "filter_expr": f"{filter_info['field']} <= {filter_info['threshold']}",
                "target_selectivity": filter_info["target_selectivity"],
                "actual_selectivity": filter_info["actual_selectivity"],
                "build_seconds": benchmark["build_seconds"],
                "median_query_seconds": benchmark["median_query_seconds"],
                "median_avg_ms_per_query": benchmark["median_avg_ms_per_query"],
                "median_qps": benchmark["median_qps"],
            }
        )

    results.sort(key=lambda item: label_order.get(item["label"], len(label_order)))
    missing_labels = [label for label in expected_labels if label not in {item["label"] for item in results}]
    baseline = results[0]

    lines = [
        "## Cross-platform Raw Hybrid Collection Benchmark Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Dataset | `{baseline['dataset_name']}` |",
        f"| Data size | `{baseline['data_num']}` |",
        f"| Dimension | `{baseline['dim']}` |",
        f"| Queries | `{baseline['query_num']}` |",
        f"| Top-k | `{baseline['topk']}` |",
        f"| ef_search | `{baseline['ef_search']}` |",
        f"| Rounds | `{baseline['rounds']}` |",
        f"| Filter | `{baseline['filter_expr']}` |",
        f"| Target selectivity | `{baseline['target_selectivity']:.4f}` |",
        "",
        "| Platform | Label | Build Seconds | Median Query Seconds | Median ms/query | Median QPS | Actual Selectivity |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for item in results:
        lines.append(
            f"| {item['platform']} | `{item['label']}` | "
            f"{item['build_seconds']:.3f} | "
            f"{item['median_query_seconds']:.3f} | "
            f"{item['median_avg_ms_per_query']:.3f} | "
            f"{item['median_qps']:.3f} | "
            f"{item['actual_selectivity']:.4f} |"
        )

    if missing_labels:
        lines.extend(["", "### Missing results", "", ", ".join(f"`{label}`" for label in missing_labels)])

    return lines


def aggregate_summary(args: argparse.Namespace) -> int:
    result_paths = sorted(args.results_dir.glob("raw-hybrid-collection-*.json"))
    expected_labels = args.expected_label or DEFAULT_LABELS
    with _summary_path().open("a", encoding="utf-8") as summary:
        summary.write("\n".join(_aggregate_summary_lines(result_paths, expected_labels)))
        summary.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-dataset")
    download.add_argument("--dataset", default=_env("BENCHMARK_DATASET", DEFAULT_DATASET))
    download.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_env("ALAYALITE_BENCH_DATA_ROOT", "./.github-tmp/alayalite-bench-data")),
    )
    download.set_defaults(func=download_dataset)

    run = subparsers.add_parser("run-benchmark")
    run.add_argument("--label", default=_env("BENCHMARK_LABEL", "local"))
    run.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(_env("ALAYALITE_BENCH_DATA_ROOT", "./.github-tmp/alayalite-bench-data")),
    )
    run.add_argument("--dataset-dir", type=Path, default=None)
    run.add_argument("--query-num", default=_env("BENCHMARK_QUERY_NUM", "100"))
    run.add_argument("--topk", default=_env("BENCHMARK_TOPK", "50"))
    run.add_argument("--ef-search", default=_env("BENCHMARK_EF_SEARCH", "100"))
    run.add_argument("--rounds", default=_env("BENCHMARK_ROUNDS", "3"))
    run.add_argument("--selectivity", default=_env("BENCHMARK_SELECTIVITY", "0.01"))
    run.add_argument("--build-threads", default=_env("BENCHMARK_BUILD_THREADS", "0"))
    run.add_argument("--query-threads", default=_env("BENCHMARK_QUERY_THREADS", "1"))
    run.add_argument("--output", type=Path, default=None)
    run.set_defaults(func=run_benchmark)

    single = subparsers.add_parser("append-summary")
    single.add_argument("--label", default=_env("BENCHMARK_LABEL", "local"))
    single.add_argument("--result", type=Path, default=None)
    single.set_defaults(func=append_summary)

    aggregate = subparsers.add_parser("aggregate-summary")
    aggregate.add_argument("--results-dir", type=Path, default=Path("artifacts/benchmarks"))
    aggregate.add_argument("--expected-label", action="append", default=None)
    aggregate.set_defaults(func=aggregate_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
