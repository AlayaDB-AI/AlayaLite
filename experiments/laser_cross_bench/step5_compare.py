#!/usr/bin/env python3
"""LASER cross-bench step5: aggregate cell CSVs into a markdown comparison table.

Auto-discovers ``<tmp_dir>/results/{orig,lite}_<tag>.csv`` for each
vamana_source in the config. Cells whose CSV is missing are skipped
(useful when you only ran a subset)::

    python step5_compare.py --config configs/gist1m.toml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CrossBenchConfig  # noqa: E402  pylint: disable=wrong-import-position

LASERS = ("orig", "lite")


def _load(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            ef = int(row["ef"])
            out[ef] = {
                "qps": float(row["qps"]),
                "recall": float(row["recall_at_10"]),
                "mean_lat": float(row["mean_lat_us"]),
                "p99_lat": float(row["p99_9_lat_us"]),
            }
    return out


def _md_table(
    data: dict[str, dict[int, dict]],
    keys: list[str],
    short: list[str],
    efs: list[int],
    field: str,
    fmt: str,
    title: str,
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| EF | " + " | ".join(short) + " |",
        "|---:|" + "|".join("---:" for _ in keys) + "|",
    ]
    for ef in efs:
        cells = []
        for k in keys:
            v = data.get(k, {}).get(ef)
            cells.append(f"{v[field]:{fmt}}" if v is not None else "—")
        lines.append(f"| {ef} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (default: <results_dir>/comparison.md)",
    )
    args = p.parse_args()

    cfg = CrossBenchConfig.from_toml(args.config)

    keys: list[str] = []
    short: list[str] = []
    legend_rows: list[tuple[str, str, str]] = []
    data: dict[str, dict[int, dict]] = {}
    for source in cfg.vamana_sources:
        for laser in LASERS:
            csv_path = cfg.cell_csv(laser, source.tag)
            if not csv_path.is_file():
                print(f"[skip] missing csv: {csv_path}")
                continue
            key = f"{laser}_{source.tag}"
            keys.append(key)
            short_label = f"{laser}+{source.tag}"
            short.append(short_label)
            legend_rows.append((short_label, "Orig Laser" if laser == "orig" else "AlayaLite", source.tag))
            data[key] = _load(csv_path)

    if not data:
        raise FileNotFoundError(f"no result CSVs found under {cfg.results_dir()} for any cell. Run step3/step4 first.")

    efs = sorted({ef for cell in data.values() for ef in cell})

    md = [f"# LASER Cross-Bench Results — {cfg.name}", ""]
    md += ["## Legend", "", "| Short | Library | Vamana Source |", "|---|---|---|"]
    for short_label, lib, vamana in legend_rows:
        md.append(f"| {short_label} | {lib} | {vamana} |")
    md.append("")
    for block in [
        _md_table(data, keys, short, efs, "qps", ".1f", "QPS"),
        [""],
        _md_table(data, keys, short, efs, "recall", ".4f", "Recall@10"),
        [""],
        _md_table(data, keys, short, efs, "mean_lat", ".1f", "Mean Latency (us)"),
        [""],
        _md_table(data, keys, short, efs, "p99_lat", ".1f", "P99.9 Latency (us)"),
    ]:
        md.extend(block)

    out = args.output or (cfg.results_dir() / "comparison.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nMarkdown written -> {out}")


if __name__ == "__main__":
    main()
