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

"""Tests for DiskCollection benchmark sweep generation."""

# pylint: disable=wrong-import-position  # importorskip must run before bench imports

import json
import sys

import pytest

pytest.importorskip("alayalite._alayalitepy", reason="bench tests require built alayalite extension")

from alayalite.bench.disk_collection import (
    _filter_ignored_args,
    _iter_sweep_points,
)
from alayalite.bench.disk_collection import (
    main as bench_main,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="DiskCollection v1 is POSIX-only")


def test_disk_flat_spec_sweep_uses_pending_axis_only():
    points = list(_iter_sweep_points("disk_flat", "spec", 10, [100], [4], [None]))
    assert len(points) == 12
    assert {p.top_k for p in points} == {1, 10, 100}
    assert {p.max_pending_bytes for p in points} == {None, 1_048_576, 262_144, 65_536}
    assert {p.ef for p in points} == {None}
    assert {p.beam_width for p in points} == {None}


def test_disk_vamana_spec_sweep_uses_ef_without_beam():
    points = list(_iter_sweep_points("disk_vamana", "spec", 10, [100], [4], [None]))
    assert len(points) == 12
    assert {p.top_k for p in points} == {1, 10, 100}
    assert {p.ef for p in points} == {50, 100, 200, 400}
    assert {p.beam_width for p in points} == {None}


def test_disk_laser_spec_sweep_is_ef_by_beam_cartesian_product():
    points = list(_iter_sweep_points("disk_laser", "spec", 10, [100], [4], [None]))
    assert len(points) == 48
    assert {p.top_k for p in points} == {1, 10, 100}
    assert {p.ef for p in points} == {50, 100, 200, 400}
    assert {p.beam_width for p in points} == {1, 2, 4, 8}


def test_ignored_args_contract():
    assert _filter_ignored_args("disk_flat", {"metric": "L2", "ef": 200, "beam_width": 4}) == [
        "ef",
        "beam_width",
    ]
    assert _filter_ignored_args("disk_vamana", {"metric": "IP", "beam_width": 4, "max_pending_bytes": None}) == [
        "beam_width",
        "metric=IP",
    ]
    assert _filter_ignored_args("disk_laser", {"metric": "COS", "max_pending_bytes": 1024}) == [
        "max_pending_bytes",
        "metric=COS",
    ]


def test_recommended_sweep_is_strict_subset_of_spec():
    """`recommended` is the curated `top_k=10` slice; `spec` is the full matrix."""
    spec = list(_iter_sweep_points("disk_flat", "spec", 10, [100], [4], [None]))
    recommended = list(_iter_sweep_points("disk_flat", "recommended", 10, [100], [4], [None]))
    assert {p.top_k for p in spec} == {1, 10, 100}
    assert {p.top_k for p in recommended} == {10}
    assert len(recommended) < len(spec)
    # Every recommended point must appear verbatim in spec (subset).
    assert set(recommended).issubset(set(spec))


def test_disk_flat_spec_sweep_round_trip_e2e(tmp_path):
    """Run the full spec sweep on disk_flat synth: every (top_k, pending) combo
    produces a distinct raw file with valid JSON, and the small
    `--max-pending-bytes` settings do NOT trip the C++ engine's 2x check
    (regression for harness vs DiskCollection.add() batch-size mismatch).
    """
    rc = bench_main(
        [
            "--engine",
            "disk_flat",
            "--dataset",
            "synth",
            "--n",
            "1024",
            "--queries",
            "32",
            "--warmup",
            "0",
            "--metric",
            "L2",
            "--seed",
            "42",
            "--out",
            str(tmp_path),
            "--run-id",
            "flat_sweep",
            "--sweep",
            "spec",
        ]
    )
    assert rc == 0
    raw_dir = tmp_path / "flat_sweep" / "raw"
    raws = sorted(raw_dir.glob("disk_flat_synth_L2_*.json"))
    # 3 top_k values × 4 max_pending_bytes values = 12 distinct files.
    assert len(raws) == 12, f"expected 12 raw files, got {[p.name for p in raws]}"
    assert len({p.name for p in raws}) == 12, "filenames must be distinct"
    # Every raw must be valid JSON with positive QPS and recall@10 ~= 1.
    for path in raws:
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["engine"] == "disk_flat"
        assert raw["dataset"] == "synth"
        assert raw["results"]["qps"] > 0
        assert raw["results"]["recall_at_10"] == pytest.approx(1.0)


def test_disk_flat_synth_sha16_differs_across_metrics(tmp_path):
    """Same (n, dim, queries, seed) under L2 vs IP must produce distinct
    `dataset_sha256_prefix` (else cross-metric runs would silently merge in
    downstream diff tools).
    """

    def run(metric: str, run_id: str) -> str:
        rc = bench_main(
            [
                "--engine",
                "disk_flat",
                "--dataset",
                "synth",
                "--n",
                "256",
                "--queries",
                "16",
                "--warmup",
                "0",
                "--metric",
                metric,
                "--seed",
                "42",
                "--out",
                str(tmp_path),
                "--run-id",
                run_id,
                "--sweep",
                "off",
            ]
        )
        assert rc == 0
        raw = json.loads(
            (tmp_path / run_id / "raw" / f"disk_flat_synth_{metric}.json").read_text(encoding="utf-8"),
        )
        return raw["provenance"]["dataset_sha256_prefix"]

    sha_l2 = run("L2", "metric_l2")
    sha_ip = run("IP", "metric_ip")
    assert sha_l2 != sha_ip, "synth dataset hash MUST differ across metrics"


def test_run_id_collision_appends_numeric_suffix(tmp_path):
    """When --run-id is reused, the second invocation creates a sibling
    directory with a 3-digit suffix instead of overwriting the first.
    """
    common = [
        "--engine",
        "disk_flat",
        "--dataset",
        "synth",
        "--n",
        "256",
        "--queries",
        "16",
        "--warmup",
        "0",
        "--metric",
        "L2",
        "--seed",
        "42",
        "--out",
        str(tmp_path),
        "--run-id",
        "fixed",
        "--sweep",
        "off",
    ]
    assert bench_main(list(common)) == 0
    assert (tmp_path / "fixed" / "summary.json").exists()
    assert bench_main(list(common)) == 0
    assert (tmp_path / "fixed_001" / "summary.json").exists()
