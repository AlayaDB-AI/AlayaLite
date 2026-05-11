#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Render recall-QPS curves from ``laser_unified_fit_bench`` JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Path to report.json from laser_unified_fit_bench")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report payload must be a JSON object: {path}")
    reports = payload.get("reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError(f"report payload missing non-empty 'reports' array: {path}")
    return payload


def _render_plot(reports: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(reports)
    fig, axes = plt.subplots(rows, 1, figsize=(8, max(4, rows * 4)), dpi=140, squeeze=False)
    style = {
        "manual": {"marker": "o", "color": "#1f77b4", "label": "Manual"},
        "unified": {"marker": "s", "color": "#d62728", "label": "Unified"},
    }

    for ax, report in zip(axes[:, 0], reports):
        ef_rows = sorted(report.get("ef_rows", []), key=lambda row: int(row["ef"]))
        if not ef_rows:
            raise ValueError(f"dataset {report.get('dataset', '<unknown>')} has empty ef_rows")

        manual_recall = [float(row["manual_recall"]) * 100.0 for row in ef_rows]
        unified_recall = [float(row["unified_recall"]) * 100.0 for row in ef_rows]
        manual_qps = [float(row["manual_qps"]) for row in ef_rows]
        unified_qps = [float(row["unified_qps"]) for row in ef_rows]
        efs = [int(row["ef"]) for row in ef_rows]

        ax.plot(manual_recall, manual_qps, linewidth=1.8, **style["manual"])
        ax.plot(unified_recall, unified_qps, linewidth=1.8, **style["unified"])

        for recall, qps, ef in zip(manual_recall, manual_qps, efs):
            ax.annotate(f"EF{ef}", (recall, qps), xytext=(5, 6), textcoords="offset points", fontsize=8)
        for recall, qps, ef in zip(unified_recall, unified_qps, efs):
            ax.annotate(f"EF{ef}", (recall, qps), xytext=(5, -12), textcoords="offset points", fontsize=8)

        dataset = report.get("dataset", "dataset")
        build_manual = float(report.get("build_manual_s", 0.0))
        build_unified = float(report.get("build_unified_s", 0.0))
        ax.set_title(f"{dataset}: Recall-QPS (build manual={build_manual:.2f}s, unified={build_unified:.2f}s)")
        ax.set_xlabel("Recall (%)")
        ax.set_ylabel("QPS")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    payload = _load_payload(args.report)
    _render_plot(payload["reports"], args.output)
    print(f"[plot] output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
