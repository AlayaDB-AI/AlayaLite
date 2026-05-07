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

"""End-to-end smoke test for the disk_vamana synth benchmark path."""

# pylint: disable=wrong-import-position  # importorskip must run before bench imports

import json
import sys

import pytest

pytest.importorskip("alayalite._alayalitepy", reason="bench tests require built alayalite extension")

from alayalite.bench.disk_collection import main as bench_main

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="DiskCollection v1 is POSIX-only")


def test_disk_vamana_synth_outputs_schema(tmp_path):
    assert (
        bench_main(
            [
                "--engine",
                "disk_vamana",
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
                "--ef",
                "128",
                "--seed",
                "42",
                "--out",
                str(tmp_path),
                "--run-id",
                "vamana",
                "--sweep",
                "off",
            ]
        )
        == 0
    )
    raw = json.loads((tmp_path / "vamana" / "raw" / "disk_vamana_synth_L2.json").read_text(encoding="utf-8"))
    assert raw["engine"] == "disk_vamana"
    assert raw["dataset"] == "synth"
    assert raw["results"]["recall_status"] == "computed"
    assert raw["results"]["recall_at_10"] >= 0.7
    assert raw["results"]["qps"] > 0
    assert raw["results"]["latency_us"]["p50"] > 0
    assert raw["results"]["on_disk_bytes"] > 0
