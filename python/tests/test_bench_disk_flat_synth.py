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

"""End-to-end smoke test for the disk_flat synth benchmark path."""

# pylint: disable=wrong-import-position  # importorskip must run before bench imports

import json
import sys

import pytest

pytest.importorskip("alayalite._alayalitepy", reason="bench tests require built alayalite extension")

from alayalite.bench.disk_collection import main as bench_main

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="DiskCollection v1 is POSIX-only")


def test_disk_flat_synth_outputs_schema(tmp_path):
    assert (
        bench_main(
            [
                "--engine",
                "disk_flat",
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
                "--metric",
                "L2",
                "--seed",
                "42",
                "--out",
                str(tmp_path),
                "--run-id",
                "flat",
                "--sweep",
                "off",
            ]
        )
        == 0
    )
    summary = json.loads((tmp_path / "flat" / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads((tmp_path / "flat" / "raw" / "disk_flat_synth_L2.json").read_text(encoding="utf-8"))
    assert "provenance" in summary
    assert summary["raws"]
    assert raw["engine"] == "disk_flat"
    assert raw["dataset"] == "synth"
    assert raw["results"]["recall_at_10"] == pytest.approx(1.0)
    assert raw["results"]["qps"] > 0
    assert raw["results"]["latency_us"]["p50"] > 0
    assert raw["results"]["on_disk_bytes"] > 0
