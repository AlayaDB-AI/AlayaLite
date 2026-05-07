#!/usr/bin/env python3
"""Plot Recall@10 vs QPS for all (laser, vamana_tag) cells in a config.

Reads CSVs from ``<tmp_dir>/results/{orig,lite}_<tag>.csv`` (no embedded data).
Cells whose CSV is missing are skipped silently::

    # Default: all cells in config (e.g. 6 curves for 3 vamana_sources × 2 lasers)
    python plot_recall_qps.py --config configs/gist1m.toml

    # Subset: only the L=100 vs L=200 comparison (mirrors the old _l200 plot)
    python plot_recall_qps.py --config configs/gist1m.toml \\
        --tags alayalite_l100 alayalite_l200

    # Only one vamana source, both lasers (mirrors the old 4-curve plot's diskann pair)
    python plot_recall_qps.py --config configs/gist1m.toml --tags diskann
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CrossBenchConfig  # noqa: E402  pylint: disable=wrong-import-position

# Stable colour palette for vamana sources; cycled if the config has more
# sources than colours. Linestyle distinguishes orig (solid) vs lite (dashed).
_PALETTE = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336", "#607D8B"]
_MARKERS = ["o", "s", "^", "D", "v", "P"]
_ANNOTATE_EFS = {80, 500}


def _load_rows(path: Path) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append((int(row["ef"]), float(row["qps"]), float(row["recall_at_10"])))
    return sorted(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Subset of vamana_source tags to plot (default: all in config)",
    )
    p.add_argument("--output", type=Path, default=None, help="Output PNG path")
    p.add_argument("--annotate-efs", type=int, nargs="*", default=sorted(_ANNOTATE_EFS))
    args = p.parse_args()

    cfg = CrossBenchConfig.from_toml(args.config)
    selected = list(cfg.vamana_sources)
    if args.tags is not None:
        wanted = set(args.tags)
        selected = [vs for vs in selected if vs.tag in wanted]
        unknown = wanted - {vs.tag for vs in cfg.vamana_sources}
        if unknown:
            raise ValueError(f"unknown vamana tags: {sorted(unknown)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    annotate_efs = set(args.annotate_efs)
    plotted = 0

    for src_idx, source in enumerate(selected):
        colour = _PALETTE[src_idx % len(_PALETTE)]
        marker = _MARKERS[src_idx % len(_MARKERS)]
        for laser, linestyle, marker_offset in (("orig", "-", 0), ("lite", "--", 4)):
            csv_path = cfg.cell_csv(laser, source.tag)
            if not csv_path.is_file():
                print(f"[skip] missing: {csv_path}")
                continue
            rows = _load_rows(csv_path)
            if not rows:
                continue
            efs = [r[0] for r in rows]
            qps = [r[1] for r in rows]
            recalls = [r[2] for r in rows]
            label = f"{('Orig Laser' if laser == 'orig' else 'AlayaLite')} + {source.tag}"
            ax.plot(
                recalls,
                qps,
                color=colour,
                linestyle=linestyle,
                marker=marker,
                markersize=6,
                linewidth=1.8,
                label=label,
            )
            for ef, recall, q in zip(efs, recalls, qps):
                if ef in annotate_efs:
                    x_offset = 4 if ef == min(annotate_efs) else -42
                    ax.annotate(
                        f"ef={ef}",
                        xy=(recall, q),
                        xytext=(x_offset, 8 + marker_offset),
                        textcoords="offset points",
                        fontsize=7.5,
                        color=colour,
                    )
            plotted += 1

    if plotted == 0:
        raise FileNotFoundError("no result CSVs found; run step3/step4 first")

    ax.set_title(
        f"{cfg.name}: Recall@10 vs QPS (LASER cross-benchmark)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Recall@10", fontsize=11)
    ax.set_ylabel("QPS (queries/sec)", fontsize=11)
    ax.grid(color="#CCCCCC", linewidth=0.6, linestyle="-", alpha=0.7)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = args.output or (cfg.results_dir() / "recall_qps.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
