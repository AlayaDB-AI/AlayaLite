#!/usr/bin/env python3
"""
Aggregate 4 CSVs into 2x2 comparison tables.

Run after all four bench scripts complete:
    python experiments/laser_cross_bench/step5_compare.py

Reads:
    $TMP/results/orig_diskann.csv
    $TMP/results/orig_lite.csv
    $TMP/results/lite_diskann.csv
    $TMP/results/lite_lite.csv

Writes:
    $TMP/results/comparison.md
"""

from __future__ import annotations

import csv
from pathlib import Path

TMP = Path("/md1/huangliang/alaya-dev/tmp/laser_cross_bench")

COMBOS = [
    ("orig_diskann", "Orig Laser", "DiskANN Vamana"),
    ("orig_lite", "Orig Laser", "Lite Vamana"),
    ("lite_diskann", "AlayaLite", "DiskANN Vamana"),
    ("lite_lite", "AlayaLite", "Lite Vamana"),
]
SHORT = ["Orig+DiskANN", "Orig+Lite", "Lite+DiskANN", "Lite+Lite"]


def load(name: str) -> dict[int, dict]:
    path = TMP / "results" / f"{name}.csv"
    out: dict[int, dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ef = int(row["ef"])
            out[ef] = {
                "qps": float(row["qps"]),
                "recall": float(row["recall_at_10"]),
                "mean_lat": float(row["mean_lat_us"]),
                "p99_lat": float(row["p99_9_lat_us"]),
            }
    return out


def _table(data: dict[str, dict[int, dict]], efs: list[int], field: str, fmt: str, title: str) -> list[str]:
    keys = [k for k, _, _ in COMBOS]
    lines = [f"### {title}", ""]
    header = f"{'EF':>5}  " + "  ".join(f"{h:>15}" for h in SHORT)
    lines += [header, "-" * len(header)]
    for ef in efs:
        vals = "  ".join(f"{data[k][ef][field]:{fmt}}" for k in keys)
        lines.append(f"{ef:>5}  {vals}")
    return lines


def _md_table(data: dict[str, dict[int, dict]], efs: list[int], field: str, fmt: str, title: str) -> list[str]:
    keys = [k for k, _, _ in COMBOS]
    lines = [
        f"### {title}",
        "",
        "| EF | " + " | ".join(SHORT) + " |",
        "|---:|" + "|".join("---:" for _ in COMBOS) + "|",
    ]
    for ef in efs:
        vals = " | ".join(f"{data[k][ef][field]:{fmt}}" for k in keys)
        lines.append(f"| {ef} | {vals} |")
    return lines


def main() -> None:
    data = {key: load(key) for key, _, _ in COMBOS}
    efs = sorted(next(iter(data.values())).keys())

    for lines in [
        _table(data, efs, "qps", ">15.1f", "QPS"),
        [""],
        _table(data, efs, "recall", ">15.4f", "Recall@10"),
        [""],
        _table(data, efs, "mean_lat", ">15.1f", "Mean Latency (us)"),
        [""],
        _table(data, efs, "p99_lat", ">15.1f", "P99.9 Latency (us)"),
    ]:
        for line in lines:
            print(line)

    md_lines = ["# LASER 2x2 Cross-Experiment Results", ""]
    md_lines += ["## Legend", "", "| Short | Library | Vamana Source |", "|---|---|---|"]
    for (_key, lib, vamana), short in zip(COMBOS, SHORT):
        md_lines.append(f"| {short} | {lib} | {vamana} |")
    md_lines.append("")
    for block in [
        _md_table(data, efs, "qps", ".1f", "QPS"),
        [""],
        _md_table(data, efs, "recall", ".4f", "Recall@10"),
        [""],
        _md_table(data, efs, "mean_lat", ".1f", "Mean Latency (us)"),
        [""],
        _md_table(data, efs, "p99_lat", ".1f", "P99.9 Latency (us)"),
    ]:
        md_lines.extend(block)

    out = TMP / "results" / "comparison.md"
    out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nMarkdown written -> {out}")


if __name__ == "__main__":
    main()
