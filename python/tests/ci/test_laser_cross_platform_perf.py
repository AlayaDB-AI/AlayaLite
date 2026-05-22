# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Static/unit checks for the LASER cross-platform perf workflow helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "laser-cross-platform-perf.yaml"
SCRIPT = ROOT / "scripts" / "laser_perf" / "laser_cross_platform_perf.py"
ALIGNED_FILE_READER = ROOT / "include" / "index" / "graph" / "laser" / "utils" / "aligned_file_reader.hpp"
LASER_IO_HEADER = ROOT / "include" / "index" / "graph" / "laser" / "utils" / "io.hpp"
LASER_MEMORY_HEADER = ROOT / "include" / "index" / "graph" / "laser" / "utils" / "memory.hpp"
LASER_ROTATOR_HEADER = ROOT / "include" / "index" / "graph" / "laser" / "utils" / "rotator.hpp"
PLATFORM_HEADER = ROOT / "include" / "utils" / "platform.hpp"
PLATFORM_FS_HEADER = ROOT / "include" / "utils" / "platform_fs.hpp"
TOOLS_HEADER = ROOT / "include" / "index" / "graph" / "laser" / "utils" / "tools.hpp"


def _load_helper():
    # Use sys.path + import (not spec_from_file_location) so coverage.py can
    # track which lines of the helper script were executed.
    script_dir = str(SCRIPT.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import laser_cross_platform_perf  # pylint: disable=import-outside-toplevel

    return laser_cross_platform_perf


def test_laser_perf_workflow_is_manual_macos_first_and_artifacted() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))

    assert "workflow_dispatch" in triggers
    # Manual-only while params / dataset are still being tuned.
    assert "pull_request" not in triggers
    assert "schedule" not in triggers
    assert "push" not in triggers
    benchmark = workflow["jobs"]["benchmark"]
    matrix = benchmark["strategy"]["matrix"]["include"]
    enabled_labels = {entry["label"] for entry in matrix if entry.get("enabled", True)}
    assert {
        "linux-libaio-x86_64",
        "macos-threadpool-arm64",
        "macos-threadpool-x86_64",
        "windows-iocp-x64",
    }.issubset(enabled_labels)
    backends_by_label = {entry["label"]: entry["backend"] for entry in matrix}
    assert backends_by_label["linux-libaio-x86_64"] == "libaio"
    assert backends_by_label["macos-threadpool-arm64"] == "threadpool"
    assert backends_by_label["macos-threadpool-x86_64"] == "threadpool"
    assert backends_by_label["windows-iocp-x64"] == "iocp"

    steps = benchmark["steps"]
    run_blocks = "\n".join(step.get("run", "") for step in steps)
    assert "brew install libomp" in run_blocks
    assert "libaio-dev" in run_blocks
    assert "laser_cross_platform_perf.py run-benchmark" in run_blocks

    upload = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v4")
    assert upload["with"]["name"] == "laser-perf-${{ matrix.label }}"
    assert "artifacts/laser-perf-${{ matrix.label }}.json" in upload["with"]["path"]
    assert "results/laser_*" in upload["with"]["path"]

    aggregate_step = next(
        step for step in workflow["jobs"]["aggregate-summary"]["steps"] if "aggregate-summary" in step.get("run", "")
    )
    assert "--baseline-label linux-libaio-x86_64" in aggregate_step["run"]


def test_laser_portable_headers_avoid_macos_compile_traps() -> None:
    aligned_reader = ALIGNED_FILE_READER.read_text(encoding="utf-8")
    laser_io = LASER_IO_HEADER.read_text(encoding="utf-8")
    laser_memory = LASER_MEMORY_HEADER.read_text(encoding="utf-8")
    laser_rotator = LASER_ROTATOR_HEADER.read_text(encoding="utf-8")
    platform = PLATFORM_HEADER.read_text(encoding="utf-8")
    platform_fs = PLATFORM_FS_HEADER.read_text(encoding="utf-8")
    tools = TOOLS_HEADER.read_text(encoding="utf-8")

    assert "#include <malloc.h>" not in aligned_reader
    assert "file_size_or" in platform_fs
    assert '#include "utils/platform_fs.hpp"' in laser_io
    assert "stat64" not in laser_io
    assert "alaya_aligned_alloc_impl" in platform
    assert '#include "utils/platform.hpp"' in laser_memory
    assert "std::aligned_alloc" not in laser_memory
    assert "propagate_on_container_move_assignment" in laser_memory
    assert "is_always_equal" in laser_memory
    assert "operator==" in laser_memory
    assert '#include "utils/platform.hpp"' in laser_rotator
    assert "#if defined(ALAYA_ARCH_X86)" in laser_rotator
    assert "third_party/ffht/fht_avx.hpp" in laser_rotator
    assert "fht_float_portable" in laser_rotator
    assert "static_cast<std::mt19937::result_type>" in tools


def test_laser_perf_summary_and_aggregate_lines(tmp_path: Path) -> None:
    helper = _load_helper()

    def _build_result(label: str, platform_info: dict, backend: str, dc_qps: float) -> dict:
        return {
            "label": label,
            "platform": platform_info,
            "backend": backend,
            "build_phase": {
                "build_wall_s": 1.5,
                "build_rss_increment_kb": 8192,
            },
            "dataset": {
                "distribution": "synthetic_gmm_l2norm",
                "n": 256,
                "dim": 128,
                "main_dim": 128,
                "seed": 42,
                "n_clusters": 8,
                "cluster_std": 0.35,
            },
            "params": {
                "queries": 8,
                "top_k": 10,
                "ef": 64,
                "beam_width": 4,
                "rounds": 2,
                "build_threads": 0,
                "query_threads": 1,
            },
            "benchmark": {
                "median_laser_api_qps": 2 * dc_qps,
                "median_disk_collection_qps": dc_qps,
                "median_dc_vs_laser_qps_ratio": 0.5,
                "median_adapter_overhead_pct": 1.5,
                "median_disk_collection_p50_us": 500.0,
                "median_disk_collection_p95_us": 800.0,
                "median_disk_collection_p99_us": 900.0,
                "median_laser_api_p50_us": 250.0,
                "median_laser_api_p95_us": 400.0,
                "median_laser_api_p99_us": 450.0,
                "median_laser_api_recall_at_10": 0.98,
                "median_disk_collection_recall_at_10": 0.97,
                "median_query_peak_rss_kb": 65536,
                "median_recall_delta": 0.0,
                "max_abs_recall_delta": 0.01,
            },
            "round_results": [
                {
                    "round": 1,
                    "laser_api_qps": 2 * dc_qps,
                    "disk_collection_qps": dc_qps,
                    "dc_vs_laser_qps_ratio": 0.5,
                    "adapter_overhead_pct": 1.5,
                    "disk_collection_p50_us": 500.0,
                    "disk_collection_p95_us": 800.0,
                    "disk_collection_p99_us": 900.0,
                    "laser_api_p50_us": 250.0,
                    "laser_api_p95_us": 400.0,
                    "laser_api_p99_us": 450.0,
                    "laser_api_recall_at_10": 0.98,
                    "disk_collection_recall_at_10": 0.97,
                    "query_peak_rss_kb": 65536,
                    "p50_delta_us": 250.0,
                    "p95_delta_us": 400.0,
                    "p99_delta_us": 450.0,
                    "recall_delta": 0.0,
                },
            ],
        }

    macos_result = _build_result(
        "macos-threadpool-arm64",
        {"system": "Darwin", "machine": "arm64", "python": "3.11.0"},
        "threadpool",
        60.0,
    )
    linux_result = _build_result(
        "linux-libaio-x86_64",
        {"system": "Linux", "machine": "x86_64", "python": "3.11.0"},
        "libaio",
        120.0,
    )

    single_lines = helper._single_summary_lines(macos_result)  # pylint: disable=protected-access
    assert "## LASER Cross-platform Benchmark (Darwin / arm64)" in single_lines
    # Header carries label/backend/dataset shape in one line.
    assert any("macos-threadpool-arm64" in line and "threadpool" in line for line in single_lines)
    # recall@10 row has `collection(native)` pair -- DC 0.97, native 0.98.
    assert any("recall@10" in line and "0.970" in line and "0.980" in line for line in single_lines)
    # QPS row: DC 60, native 120 (in `dc_qps * 2` pattern).
    assert any(line.startswith("| QPS |") and "60.0" in line and "120.0" in line for line in single_lines)
    # Latency p50/p95/p99 expressed in ms with collection(native).
    assert any("p50 (ms)" in line and "0.50" in line and "0.25" in line for line in single_lines)
    assert any("p95 (ms)" in line and "0.80" in line and "0.40" in line for line in single_lines)
    assert any("p99 (ms)" in line and "0.90" in line and "0.45" in line for line in single_lines)
    # Build + query memory rows.
    assert any("build wall (s)" in line and "1.5" in line for line in single_lines)
    assert any("build RSS" in line and "8.0" in line for line in single_lines)  # 8192 KB = 8.0 MB
    assert any("query peak RSS" in line and "64.0" in line for line in single_lines)  # 65536 KB = 64.0 MB

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    macos_path = artifact_dir / "laser-perf-macos-threadpool-arm64.json"
    linux_path = artifact_dir / "laser-perf-linux-libaio-x86_64.json"
    macos_path.write_text(json.dumps(macos_result), encoding="utf-8")
    linux_path.write_text(json.dumps(linux_result), encoding="utf-8")
    result_paths = helper._aggregate_result_paths(tmp_path)  # pylint: disable=protected-access
    assert set(result_paths) == {macos_path, linux_path}
    aggregate_lines = helper._aggregate_summary_lines(  # pylint: disable=protected-access
        result_paths,
        ["linux-libaio-x86_64", "macos-threadpool-arm64", "macos-threadpool-x86_64"],
        "linux-libaio-x86_64",
    )
    assert "## LASER Cross-platform Benchmark Summary" in aggregate_lines
    assert any("`linux-libaio-x86_64`" in line and "`libaio`" in line for line in aggregate_lines)
    assert any("`macos-threadpool-arm64`" in line and "`threadpool`" in line for line in aggregate_lines)
    assert any("`macos-threadpool-x86_64`" in line for line in aggregate_lines)
    # Each row carries collection(native) pairs for recall + QPS + latency.
    # macos: dc_qps=60, native_qps=120, dc_p50=500us=0.50ms, native_p50=250us=0.25ms.
    assert any("`macos-threadpool-arm64`" in line and "60.0" in line and "120.0" in line for line in aggregate_lines)
    assert any("`macos-threadpool-arm64`" in line and "0.50" in line and "0.25" in line for line in aggregate_lines)
    # Build phase + query RSS columns surface in every row.
    assert any("`macos-threadpool-arm64`" in line and "8.0" in line and "64.0" in line for line in aggregate_lines)


def test_generate_vectors_shape_and_l2_norm_and_reproducibility() -> None:
    """GMM data generator: shape, unit-norm, byte-identical replay under same seed."""
    helper = _load_helper()
    vectors_a = helper._generate_vectors(  # pylint: disable=protected-access
        n=512, dim=32, seed=42, n_clusters=8, cluster_std=0.3
    )
    assert vectors_a.dtype == np.float32
    assert vectors_a.shape == (512, 32)
    norms = np.linalg.norm(vectors_a, axis=1)
    np.testing.assert_allclose(norms, np.ones(512), atol=1e-5)

    # PCG64 is platform-independent — same seed must reproduce byte-identically.
    vectors_b = helper._generate_vectors(  # pylint: disable=protected-access
        n=512, dim=32, seed=42, n_clusters=8, cluster_std=0.3
    )
    np.testing.assert_array_equal(vectors_a, vectors_b)

    # Different seed → different draws.
    vectors_c = helper._generate_vectors(  # pylint: disable=protected-access
        n=512, dim=32, seed=43, n_clusters=8, cluster_std=0.3
    )
    assert not np.array_equal(vectors_a, vectors_c)


def test_detect_backend_branches(monkeypatch) -> None:
    helper = _load_helper()
    # Explicit override bypasses platform sniffing.
    assert helper._detect_backend("libaio") == "libaio"  # pylint: disable=protected-access
    assert helper._detect_backend("threadpool") == "threadpool"  # pylint: disable=protected-access
    # Auto path: Darwin → threadpool, Linux x86_64 → libaio, else unsupported.
    monkeypatch.setattr(helper.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(helper.platform, "machine", lambda: "arm64")
    assert helper._detect_backend("auto") == "threadpool"  # pylint: disable=protected-access
    monkeypatch.setattr(helper.platform, "system", lambda: "Linux")
    monkeypatch.setattr(helper.platform, "machine", lambda: "x86_64")
    assert helper._detect_backend("auto") == "libaio"  # pylint: disable=protected-access
    monkeypatch.setattr(helper.platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(helper.platform, "machine", lambda: "x86_64")
    assert helper._detect_backend("auto") == "unsupported"  # pylint: disable=protected-access


def test_median_and_fmt_helpers() -> None:
    helper = _load_helper()
    # _median: None entries filtered; empty/all-None → None; sorts numerically.
    assert helper._median([]) is None  # pylint: disable=protected-access
    assert helper._median([None, None]) is None  # pylint: disable=protected-access
    assert helper._median([1.0, 3.0, 5.0]) == 3.0  # pylint: disable=protected-access
    assert helper._median([1.0, None, 5.0]) == 3.0  # pylint: disable=protected-access

    # _fmt: None → "n/a", floats honour digits, others stringified.
    assert helper._fmt(None) == "n/a"  # pylint: disable=protected-access
    assert helper._fmt(1.23456) == "1.235"  # pylint: disable=protected-access
    assert helper._fmt(1.23456, digits=4) == "1.2346"  # pylint: disable=protected-access
    assert helper._fmt(42) == "42"  # pylint: disable=protected-access


def test_parser_defaults_aligned_with_gist1m_recipe() -> None:
    """Paper-aligned defaults must not regress silently (gist1m_run01.toml)."""
    helper = _load_helper()
    parser = helper.build_parser()
    args = parser.parse_args(["run-benchmark"])
    # Build params from paper recipe.
    assert args.degree == 64
    assert args.build_l == 200
    assert args.alpha == 1.2
    assert args.ef_indexing == 200
    assert args.ep_num == 300
    # Search params.
    assert args.beam_width == 16
    assert args.warmup == 2
    assert args.ef_search == 200
    assert args.topk == 10
    # v1 binding constraints.
    assert args.dim == 256
    assert args.main_dim == 256
    assert args.search_dram_budget_gb == 0.5  # locked by laser_compare v1 binding
    # GMM dataset.
    assert args.n_clusters == 64
    assert args.cluster_std == 0.35


def test_build_result_aggregates_rounds_and_records_recipe() -> None:
    helper = _load_helper()
    parser = helper.build_parser()
    args = parser.parse_args(
        [
            "run-benchmark",
            "--label",
            "test-label",
            "--n",
            "100",
            "--dim",
            "32",
            "--main-dim",
            "32",
        ]
    )
    summaries = [
        {
            "comparison": {
                "qps_native": 100.0,
                "qps_disk_laser": 50.0,
                "qps_ratio": 0.5,
                "adapter_overhead_pct": 1.0,
                "recall_delta": 0.0,
                "latency_us": {
                    "native": {"p50": 200, "p95": 300, "p99": 400},
                    "disk_laser": {"p50": 400, "p95": 600, "p99": 700},
                },
                "latency_us_delta": {"p50": 200, "p95": 300, "p99": 300},
            }
        },
        {
            "comparison": {
                "qps_native": 200.0,
                "qps_disk_laser": 80.0,
                "qps_ratio": 0.4,
                "adapter_overhead_pct": 2.0,
                "recall_delta": 0.01,
                "latency_us": {
                    "native": {"p50": 100, "p95": 200, "p99": 250},
                    "disk_laser": {"p50": 250, "p95": 400, "p99": 500},
                },
                "latency_us_delta": {"p50": 150, "p95": 200, "p99": 250},
            }
        },
        {
            "comparison": {
                "qps_native": 150.0,
                "qps_disk_laser": 60.0,
                "qps_ratio": 0.4,
                "adapter_overhead_pct": 1.5,
                "recall_delta": -0.02,
                "latency_us": {
                    "native": {"p50": 150, "p95": 250, "p99": 300},
                    "disk_laser": {"p50": 300, "p95": 500, "p99": 600},
                },
                "latency_us_delta": {"p50": 150, "p95": 250, "p99": 300},
            }
        },
    ]
    build_stats = {"vectors_path": Path("/tmp/fake.fbin"), "build_wall_s": 1.5, "build_rss_increment_kb": 8192}
    result = helper._build_result(args, summaries, "threadpool", build_stats)  # pylint: disable=protected-access
    # Build phase surfaces alongside the search-phase benchmark medians.
    assert result["build_phase"]["build_wall_s"] == 1.5
    assert result["build_phase"]["build_rss_increment_kb"] == 8192
    # Medians.
    assert result["benchmark"]["median_laser_api_qps"] == 150.0
    assert result["benchmark"]["median_disk_collection_qps"] == 60.0
    # Latency medians surface so cross-platform trend can compare actual us numbers.
    assert result["benchmark"]["median_disk_collection_p50_us"] == 300
    assert result["benchmark"]["median_disk_collection_p95_us"] == 500
    assert result["benchmark"]["median_disk_collection_p99_us"] == 600
    # max_abs_recall_delta = max(|0.0|, |0.01|, |-0.02|) = 0.02
    assert result["benchmark"]["max_abs_recall_delta"] == 0.02
    # Recipe fields surface through dataset + params so trend artifacts stay interpretable.
    assert result["dataset"]["distribution"] == "synthetic_gmm_l2norm"
    assert result["dataset"]["n_clusters"] == 64
    assert result["params"]["ep_num"] == 300
    assert result["params"]["build_l"] == 200
    assert result["params"]["search_dram_budget_gb"] == 0.5
    assert result["round_results"][0]["round"] == 1
    assert len(result["round_results"]) == 3


def test_write_result_files_creates_expected_artifacts(tmp_path: Path) -> None:
    helper = _load_helper()
    result = {
        "label": "macos-threadpool-arm64",
        "platform": {"system": "Darwin", "machine": "arm64", "python": "3.11.0"},
        "backend": "threadpool",
        "dataset": {
            "distribution": "synthetic_gmm_l2norm",
            "n": 128,
            "dim": 32,
            "main_dim": 32,
            "seed": 42,
            "n_clusters": 4,
            "cluster_std": 0.3,
        },
        "params": {
            "queries": 4,
            "top_k": 10,
            "ef": 64,
            "beam_width": 16,
            "rounds": 1,
            "warmup": 2,
            "build_threads": 0,
            "query_threads": 1,
            "degree": 64,
            "build_l": 200,
            "alpha": 1.2,
            "ef_indexing": 200,
            "ep_num": 300,
            "build_dram_budget_gb": 4.0,
            "search_dram_budget_gb": 0.5,
        },
        "benchmark": {
            "median_laser_api_qps": 10.0,
            "median_disk_collection_qps": 5.0,
            "median_dc_vs_laser_qps_ratio": 0.5,
            "median_adapter_overhead_pct": 1.5,
            "median_disk_collection_p50_us": 300.0,
            "median_disk_collection_p95_us": 500.0,
            "median_disk_collection_p99_us": 600.0,
            "median_laser_api_p50_us": 150.0,
            "median_laser_api_p95_us": 250.0,
            "median_recall_delta": 0.0,
            "max_abs_recall_delta": 0.0,
        },
        "round_results": [
            {
                "round": 1,
                "laser_api_qps": 10.0,
                "disk_collection_qps": 5.0,
                "dc_vs_laser_qps_ratio": 0.5,
                "adapter_overhead_pct": 1.5,
                "disk_collection_p50_us": 300.0,
                "disk_collection_p95_us": 500.0,
                "disk_collection_p99_us": 600.0,
                "recall_delta": 0.0,
            },
        ],
    }
    output_path = tmp_path / "artifact.json"
    markdown_path = tmp_path / "artifact.md"
    results_dir = tmp_path / "results"
    helper._write_result_files(result, output_path, markdown_path, results_dir)  # pylint: disable=protected-access

    assert output_path.is_file()
    assert markdown_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8"))["label"] == "macos-threadpool-arm64"
    # Label-slug mirror under results/ for the github-actions upload step.
    assert (results_dir / "laser_macos_threadpool_arm64_perf.json").is_file()
    assert (results_dir / "laser_macos_threadpool_arm64_perf.md").is_file()
    # macOS labels also written to a stable smoke filename for trend stitching.
    assert (results_dir / "laser_macos_smoke.json").is_file()
