# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for disk_laser runtime gating in the benchmark harness."""

import json

import numpy as np
from alayalite.bench import _engines, disk_collection
from alayalite.bench._datasets import DatasetSpec
from alayalite.bench.disk_collection import main as bench_main


def test_disk_laser_unsupported_probe_skips_cleanly(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(disk_collection, "probe_disk_laser_supported", lambda scratch_root: False)

    argv = [
        "--engine",
        "disk_laser",
        "--dataset",
        "synth",
        "--n",
        "1024",
        "--queries",
        "64",
        "--warmup",
        "8",
        "--out",
        str(tmp_path),
        "--run-id",
        "laser_skip",
    ]
    assert bench_main(argv) == 0
    assert "engine=disk_laser status=skipped reason=probe_failed" in capsys.readouterr().out
    raw = json.loads((tmp_path / "laser_skip" / "raw" / "disk_laser_synth_L2_skip.json").read_text(encoding="utf-8"))
    assert raw["status"] == "skipped"
    assert raw["reason"] == "probe_failed"
    assert raw["results"]["recall_status"] == "skipped"


def _fake_measure_search(col, dataset, params, search):
    """Stub replacement for `_engines._measure_search` — fixed QPS/latency."""
    del col, dataset, params, search
    return (
        {"qps": 1.0, "latency_us": {"p50": 1.0, "p95": 1.0, "p99": 1.0, "min": 1.0, "mean": 1.0}},
        [[0, 1, 2, 3], [0, 1, 2, 3]],
    )


def test_disk_laser_synth_recall_is_skipped_for_external_artifacts(tmp_path, monkeypatch):
    dataset = DatasetSpec(
        name="synth",
        vectors=np.zeros((4, 2), dtype=np.float32),
        labels=np.arange(4, dtype=np.uint64),
        queries=np.zeros((2, 2), dtype=np.float32),
        ground_truth=np.zeros((2, 4), dtype=np.uint64),
        sha16="0123456789abcdef",
    )

    monkeypatch.setattr(_engines, "_build_disk_laser", lambda col_path, dataset, src_dir: (object(), 0.01))
    monkeypatch.setattr(_engines, "_measure_search", _fake_measure_search)

    result = _engines.bench_disk_laser(
        dataset,
        {
            "laser_src_dir": str(tmp_path),
            "laser_recall_valid": False,
            "queries": 2,
            "warmup": 0,
            "top_k": 4,
            "ef": 50,
            "beam_width": 1,
            "scratch_root": str(tmp_path),
        },
    )
    assert result["recall_status"] == "skipped"
    assert result["recall_at_10"] is None
