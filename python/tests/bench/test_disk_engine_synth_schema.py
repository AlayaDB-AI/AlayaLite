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

"""End-to-end schema smoke for in-memory disk_flat / disk_vamana paths.

Single parametrized test consolidating what were previously two
near-duplicate modules — the only differences are the engine name,
the recall floor (``disk_flat`` is exact at recall@10 == 1.0;
``disk_vamana`` is approximate at ≥ 0.7), and the engine-specific
``--ef`` flag.
"""

import json

import pytest
from alayalite.bench.disk_collection import main as bench_main


@pytest.mark.parametrize(
    "engine, extra_argv, recall_assertion",
    [
        ("disk_flat", [], lambda r: r == pytest.approx(1.0)),
        ("disk_vamana", ["--ef", "128"], lambda r: r >= 0.7),
    ],
)
def test_engine_synth_outputs_schema(tmp_path, engine, extra_argv, recall_assertion):
    run_id = engine
    argv = [
        "--engine",
        engine,
        "--dataset",
        "synth",
        "--metric",
        "L2",
        "--n",
        "1024",
        "--queries",
        "64",
        "--warmup",
        "8",
        *extra_argv,
        "--out",
        str(tmp_path),
        "--run-id",
        run_id,
        "--sweep",
        "off",
    ]
    assert bench_main(argv) == 0

    raw_path = tmp_path / run_id / "raw" / f"{engine}_synth_L2.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw["engine"] == engine
    assert raw["dataset"] == "synth"
    assert raw["results"]["recall_status"] == "computed"
    assert recall_assertion(raw["results"]["recall_at_10"])
    assert raw["results"]["qps"] > 0
    assert raw["results"]["latency_us"]["p50"] > 0
    assert raw["results"]["on_disk_bytes"] > 0

    # disk_flat path additionally checks the top-level summary shape; one
    # of the parametrized cases is enough to anchor the summary schema.
    if engine == "disk_flat":
        summary = json.loads((tmp_path / run_id / "summary.json").read_text(encoding="utf-8"))
        assert "provenance" in summary
        assert summary["raws"]
