# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Smoke test for ``python/benchmarks/laser_unified_fit_plot.py``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_plot_cli_writes_curve_png(tmp_path: Path) -> None:
    report = {
        "schema_version": 1,
        "reports": [
            {
                "dataset": "gist1m",
                "build_manual_s": 10.0,
                "build_unified_s": 9.5,
                "ef_rows": [
                    {
                        "ef": 100,
                        "manual_recall": 0.90,
                        "unified_recall": 0.91,
                        "manual_qps": 5000.0,
                        "unified_qps": 4800.0,
                    },
                    {
                        "ef": 200,
                        "manual_recall": 0.95,
                        "unified_recall": 0.955,
                        "manual_qps": 4200.0,
                        "unified_qps": 3900.0,
                    },
                ],
            }
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    out_path = tmp_path / "curve.png"

    completed = subprocess.run(
        [
            sys.executable,
            "python/benchmarks/laser_unified_fit_plot.py",
            "--report",
            str(report_path),
            "--output",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, (
        f"plot CLI exited {completed.returncode}: stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert out_path.is_file(), f"missing {out_path}"
    assert out_path.stat().st_size > 0, f"empty {out_path}"
