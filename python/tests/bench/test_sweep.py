# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for DiskCollection benchmark sweep generation."""

import json

import pytest
from alayalite.bench.disk_collection import (
    _filter_ignored_args,
    _iter_sweep_points,
)
from alayalite.bench.disk_collection import (
    main as bench_main,
)


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
    # Mixed format: bare key means the whole flag is ignored for this engine;
    # ``key=value`` means only this specific value is unsupported.
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
    assert set(recommended).issubset(set(spec))


def _bench_synth_argv(*, tmp_path, run_id, sweep="off", **overrides) -> list[str]:
    """Build a ``bench_main`` argv for an in-memory disk_flat / synth run.

    Only flags that differ from argparse defaults are explicit; ``overrides``
    lets a caller plug in extras (``metric=...``, ``n=...``) without
    re-specifying the common shape.
    """
    base = {
        "engine": overrides.pop("engine", "disk_flat"),
        "dataset": "synth",
        "n": "256",
        "queries": "16",
        "warmup": "0",
        "out": str(tmp_path),
        "run-id": run_id,
        "sweep": sweep,
    }
    base.update({k.replace("_", "-"): str(v) for k, v in overrides.items()})
    return [tok for k, v in base.items() for tok in (f"--{k}", v)]


def test_disk_flat_spec_sweep_round_trip_e2e(tmp_path):
    """Run the full spec sweep on disk_flat synth: every (top_k, pending) combo
    produces a distinct raw file with valid JSON, and the small
    `--max-pending-bytes` settings do NOT trip the C++ engine's 2x check
    (regression for harness vs DiskCollection.add() batch-size mismatch).
    """
    argv = _bench_synth_argv(
        tmp_path=tmp_path,
        run_id="flat_sweep",
        sweep="spec",
        n=1024,
        queries=32,
    )
    assert bench_main(argv) == 0
    raw_dir = tmp_path / "flat_sweep" / "raw"
    raws = sorted(raw_dir.glob("disk_flat_synth_L2_*.json"))
    # 3 top_k values × 4 max_pending_bytes values = 12 distinct files.
    assert len(raws) == 12, f"expected 12 raw files, got {[p.name for p in raws]}"
    assert len({p.name for p in raws}) == 12, "filenames must be distinct"
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

    def sha_for(metric: str, run_id: str) -> str:
        assert bench_main(_bench_synth_argv(tmp_path=tmp_path, run_id=run_id, metric=metric)) == 0
        raw = json.loads(
            (tmp_path / run_id / "raw" / f"disk_flat_synth_{metric}.json").read_text(encoding="utf-8"),
        )
        return raw["provenance"]["dataset_sha256_prefix"]

    assert sha_for("L2", "metric_l2") != sha_for("IP", "metric_ip"), "synth dataset hash MUST differ across metrics"


def test_run_id_collision_appends_numeric_suffix(tmp_path):
    """When --run-id is reused, the second invocation creates a sibling
    directory with a 3-digit suffix instead of overwriting the first.
    """
    argv = _bench_synth_argv(tmp_path=tmp_path, run_id="fixed")
    assert bench_main(argv) == 0
    assert (tmp_path / "fixed" / "summary.json").exists()
    assert bench_main(argv) == 0
    assert (tmp_path / "fixed_001" / "summary.json").exists()
