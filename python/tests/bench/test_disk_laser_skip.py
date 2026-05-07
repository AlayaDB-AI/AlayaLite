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

"""Tests for disk_laser runtime gating in the benchmark harness."""

# pylint: disable=wrong-import-position  # importorskip must run before bench imports

import json
import sys

import numpy as np
import pytest

pytest.importorskip("alayalite._alayalitepy", reason="bench tests require built alayalite extension")

from alayalite.bench import _engines
from alayalite.bench._datasets import DatasetSpec
from alayalite.bench._engines import probe_disk_laser_supported
from alayalite.bench.disk_collection import main as bench_main

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="DiskCollection v1 is POSIX-only")


def test_disk_laser_unsupported_probe_skips_cleanly(tmp_path, capsys):
    if probe_disk_laser_supported():
        pytest.skip("disk_laser is supported on this build")

    assert (
        bench_main(
            [
                "--engine",
                "disk_laser",
                "--dataset",
                "synth",
                "--n",
                "1024",
                "--queries",
                "64",
                "--k",
                "10",
                "--warmup",
                "8",
                "--out",
                str(tmp_path),
                "--run-id",
                "laser_skip",
            ]
        )
        == 0
    )
    assert "engine=disk_laser status=skipped reason=probe_failed" in capsys.readouterr().out
    raw = json.loads((tmp_path / "laser_skip" / "raw" / "disk_laser_synth_L2_skip.json").read_text(encoding="utf-8"))
    assert raw["status"] == "skipped"
    assert raw["reason"] == "probe_failed"
    assert raw["results"]["recall_status"] == "skipped"


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
    monkeypatch.setattr(
        _engines,
        "_measure_search",
        lambda col, dataset, params, search: (
            {"qps": 1.0, "latency_us": {"p50": 1.0, "p95": 1.0, "p99": 1.0, "min": 1.0, "mean": 1.0}},
            [[0, 1, 2, 3], [0, 1, 2, 3]],
        ),
    )

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
