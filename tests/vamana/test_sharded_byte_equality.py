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

"""Tier A partition-stage byte-equality gate (`diskann-sharded-alignment-gate`).

Builds the same partition-merge Vamana graph twice — once with AlayaLite's
`build_vamana_index` CLI, once with the patched upstream DiskANN
`build_merged_vamana_standalone` CLI — at matched seeds and asserts the
**partition-stage** invariants the alignment patches pin down:

  1. `_medoids.bin` — strict byte-equality (partition-stage proof; the
     file is fully determined by per-shard point assignment)
  2. `num_parts` — structural parity (both pipelines see the same
     growth-loop trajectory)
  3. `_centroids.bin` and merged `.index` — header parity only
     (file size, dim/R/start-node fields)

Per-shard `_subshard-<i>_mem.index` byte-equality is **not** required —
that requires aligning the inner Vamana build (frozen-point semantics,
`visit_order` permutation, Lloyd-iteration count), which is deferred
to a future `align-diskann-vamana-build` change. See
`openspec/changes/align-diskann-sharded-with-upstream/design.md`
"Discovered limitation" for the full rationale.

The patched DiskANN binary lives on the `align-diskann-sharded-with-alaya`
branch in `/md1/huangliang/alaya-dev/Laser/DiskANN`. Its alignment kwargs
(`--seed`, `--shuffle_seed`, `--drop_self_loops`,
`--forced_global_medoid`, `--keep_intermediates`) opt the build into
AlayaLite-compatible RNG + merge semantics.

Skip behavior:
  * Missing dataset → skip with `synth_100k_512d` reason.
  * `DISKANN_ALIGNED_BIN` unset and the binary unfindable → skip with a
    pointer to the upstream branch.

Fixture regeneration:
  rm tests/vamana/fixtures/sharded_synth_100k_512d_M0p05.json
  pytest tests/vamana/test_sharded_byte_equality.py::test_sharded_generates_baseline
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

SYNTH_DATASET = Path("/md1/huangliang/alaya-dev/data/synth_100k_512d")
SYNTH_BASE_FBIN = SYNTH_DATASET / "base.fbin"
GIST_DATASET = Path("/md1/huangliang/alaya-dev/data/gist1m")
GIST_BASE_FBIN = GIST_DATASET / "gist_base.fbin"

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# Phase 4.2 — locked budget value. 0.05 GiB on synth_100k_512d with
# auto sampling_rate (`min(1.0, 256000/N)` = 1.0 here) drives both
# pipelines into the partition path with `num_parts = 13`. Wall time
# at `-T 1` is ~3-4 minutes per build (~7 min total for the test).
SYNTH_BUDGET_GIB = "0.05"
SYNTH_FIXTURE = FIXTURE_DIR / "sharded_synth_100k_512d_M0p05.json"
GIST_BUDGET_GIB = "0.5"
GIST_FIXTURE = FIXTURE_DIR / "sharded_gist1m_M0p5.json"

# Pinned alignment parameters (spec scenario "synth_100k_512d Tier A").
PINNED_R = "64"
PINNED_L = "100"
PINNED_ALPHA = "1.2"
PINNED_SEED = "1234"
PINNED_THREADS = "1"

DEFAULT_DISKANN_BIN = Path("/md1/huangliang/alaya-dev/Laser/DiskANN/build/apps/build_merged_vamana_standalone")


def _alaya_cli() -> Path | None:
    candidates = [
        REPO_ROOT / "build" / "tools" / "build_vamana_index" / "build_vamana_index",
        REPO_ROOT / "build-release" / "tools" / "build_vamana_index" / "build_vamana_index",
    ]
    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return path
    return None


def _diskann_aligned_bin() -> Path | None:
    env_path = os.environ.get("DISKANN_ALIGNED_BIN")
    if env_path:
        path = Path(env_path)
        return path if path.exists() and os.access(path, os.X_OK) else None
    if DEFAULT_DISKANN_BIN.exists() and os.access(DEFAULT_DISKANN_BIN, os.X_OK):
        return DEFAULT_DISKANN_BIN
    return None


def _read_merged_header(merged_index_path: Path) -> dict:
    """Parse the 24-byte merged `.index` header.

    Layout (little-endian): u64 expected_file_size, u32 R, u32 medoid,
    u64 frozen_pts. Matches AlayaLite shard_merger and DiskANN merge_shards.
    """
    with merged_index_path.open("rb") as fh:
        header = fh.read(24)
    if len(header) != 24:
        raise RuntimeError(f"merged .index header truncated: {merged_index_path}")
    expected_size, R, medoid, frozen_pts = struct.unpack("<QIIQ", header)
    return {
        "expected_size": int(expected_size),
        "R": int(R),
        "medoid": int(medoid),
        "frozen_pts": int(frozen_pts),
        "actual_size": merged_index_path.stat().st_size,
    }


def _read_centroids_header(centroids_path: Path) -> dict:
    """Parse the 8-byte centroids.bin header.

    Layout (little-endian): i32 num_parts, i32 dim. Matches both
    `diskann::save_bin<float>` and AlayaLite's centroids writer.
    """
    with centroids_path.open("rb") as fh:
        header = fh.read(8)
    if len(header) != 8:
        raise RuntimeError(f"centroids.bin header truncated: {centroids_path}")
    num_parts, dim = struct.unpack("<ii", header)
    return {
        "num_parts": int(num_parts),
        "dim": int(dim),
        "actual_size": centroids_path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_alaya(cli: Path, data_path: Path, prefix: Path, budget_gib: str) -> None:
    subprocess.run(
        [
            str(cli),
            "--data_path",
            str(data_path),
            "--index_path_prefix",
            str(prefix),
            "-R",
            PINNED_R,
            "-L",
            PINNED_L,
            "--alpha",
            PINNED_ALPHA,
            "--seed",
            PINNED_SEED,
            "-T",
            PINNED_THREADS,
            "--build_dram_budget",
            budget_gib,
        ],
        check=True,
        env={**os.environ, "OMP_NUM_THREADS": PINNED_THREADS},
    )


def _run_diskann_aligned(
    cli: Path,
    data_path: Path,
    prefix: Path,
    budget_gib: str,
    forced_medoid: int,
) -> None:
    subprocess.run(
        [
            str(cli),
            "--data_type",
            "float",
            "--dist_fn",
            "l2",
            "--data_path",
            str(data_path),
            "--index_path_prefix",
            str(prefix),
            "-R",
            PINNED_R,
            "-L",
            PINNED_L,
            "-T",
            PINNED_THREADS,
            "-M",
            budget_gib,
            "--seed",
            PINNED_SEED,
            "--shuffle_seed",
            PINNED_SEED,
            "--drop_self_loops",
            "--forced_global_medoid",
            str(forced_medoid),
            "--keep_intermediates",
        ],
        check=True,
        env={**os.environ, "OMP_NUM_THREADS": PINNED_THREADS},
    )


def _alaya_artifact_paths(prefix: Path) -> dict[str, Path | list[Path]]:
    work_dir = prefix.parent / (prefix.name + "_shard_work")
    per_shard = sorted(
        work_dir.glob("s_subshard-*_mem.index"),
        key=lambda p: int(p.stem.split("-")[1].split("_")[0]),
    )
    return {
        "merged_index": prefix,
        "centroids": Path(str(prefix) + "_centroids.bin"),
        "medoids": Path(str(prefix) + "_medoids.bin"),
        "per_shard": per_shard,
    }


def _diskann_artifact_paths(prefix: Path) -> dict[str, Path | list[Path]]:
    merged = Path(str(prefix) + ".index")
    work_root = merged.parent
    work_prefix = merged.name + "_tempFiles_subshard-"
    per_shard = sorted(
        work_root.glob(work_prefix + "*_mem.index"),
        key=lambda p: int(p.name.removeprefix(work_prefix).split("_")[0]),
    )
    return {
        "merged_index": merged,
        "centroids": Path(str(prefix) + "_centroids.bin"),
        "medoids": Path(str(prefix) + "_medoids.bin"),
        "per_shard": per_shard,
    }


def _gather_metrics(arts: dict[str, Path | list[Path]]) -> dict:
    """Spec-pinned metrics: medoid SHA + structural fields."""
    return {
        "medoids_sha256": _sha256(arts["medoids"]),
        "merged_header": _read_merged_header(arts["merged_index"]),
        "centroids_header": _read_centroids_header(arts["centroids"]),
        "num_shards": len(arts["per_shard"]),
        # Per-shard SHAs are recorded for diagnostic purposes only — they
        # are not asserted equal between AlayaLite and DiskANN. See the
        # "Discovered limitation" entry in design.md.
        "per_shard_sha256": [_sha256(p) for p in arts["per_shard"]],
        "per_shard_size": [p.stat().st_size for p in arts["per_shard"]],
    }


def _run_pair(
    alaya_cli: Path,
    diskann_cli: Path,
    data_path: Path,
    budget_gib: str,
    out_dir: Path,
) -> tuple[dict, dict]:
    alaya_prefix = out_dir / "alaya"
    diskann_prefix = out_dir / "diskann"
    _run_alaya(alaya_cli, data_path, alaya_prefix, budget_gib)
    medoid = _read_merged_header(alaya_prefix)["medoid"]
    _run_diskann_aligned(diskann_cli, data_path, diskann_prefix, budget_gib, medoid)
    a_arts = _alaya_artifact_paths(alaya_prefix)
    d_arts = _diskann_artifact_paths(diskann_prefix)
    return _gather_metrics(a_arts), _gather_metrics(d_arts)


def _check_partition_alignment(alaya: dict, diskann: dict) -> tuple[bool, str]:
    """Spec-pinned partition-stage assertions. Returns (passed, message)."""
    if alaya["medoids_sha256"] != diskann["medoids_sha256"]:
        return False, (
            f"_medoids.bin SHA mismatch — partition-stage divergence:\n"
            f"  alaya  = {alaya['medoids_sha256']}\n"
            f"  diskann= {diskann['medoids_sha256']}\n"
            f"This is the partition-stage byte-equality assertion. Investigate "
            f"the seeded RNG plumbing in DiskANN partition.cpp before looking "
            f"at downstream artifacts."
        )
    if alaya["num_shards"] != diskann["num_shards"]:
        return False, (f"num_parts mismatch: alaya={alaya['num_shards']} diskann={diskann['num_shards']}")
    if alaya["centroids_header"]["num_parts"] != diskann["centroids_header"]["num_parts"]:
        return False, "centroids.bin num_parts header divergence"
    if alaya["centroids_header"]["dim"] != diskann["centroids_header"]["dim"]:
        return False, "centroids.bin dim header divergence"
    if alaya["centroids_header"]["actual_size"] != diskann["centroids_header"]["actual_size"]:
        return False, "centroids.bin file size divergence"
    for field in ("R", "medoid", "frozen_pts"):
        if alaya["merged_header"][field] != diskann["merged_header"][field]:
            return False, (
                f"merged .index {field} header divergence: "
                f"alaya={alaya['merged_header'][field]} "
                f"diskann={diskann['merged_header'][field]}"
            )
    return True, "partition-stage byte-equality verified"


def _alignment_metrics_for_check() -> dict:
    return {
        "medoids_sha256": "same",
        "num_shards": 13,
        "centroids_header": {
            "num_parts": 13,
            "dim": 512,
            "actual_size": 26632,
        },
        "merged_header": {
            "R": 64,
            "medoid": 123,
            "frozen_pts": 0,
        },
    }


@pytest.mark.parametrize(
    ("field", "diskann_value", "expected_msg"),
    [
        ("medoid", 456, "merged .index medoid header divergence"),
        ("frozen_pts", 1, "merged .index frozen_pts header divergence"),
    ],
)
def test_partition_alignment_rejects_merged_start_header_mismatch(
    field: str, diskann_value: int, expected_msg: str
) -> None:
    """Tier A must catch merged start-header drift, not just partition files."""
    alaya = _alignment_metrics_for_check()
    diskann = _alignment_metrics_for_check()
    diskann["merged_header"] = {
        **diskann["merged_header"],
        field: diskann_value,
    }

    ok, msg = _check_partition_alignment(alaya, diskann)

    assert not ok
    assert expected_msg in msg


@pytest.fixture(scope="module")
def alaya_cli() -> Path:
    binary = _alaya_cli()
    if binary is None:
        pytest.skip("AlayaLite build_vamana_index missing; run cmake --build build --target build_vamana_index")
    return binary


@pytest.fixture(scope="module")
def diskann_aligned_cli() -> Path:
    binary = _diskann_aligned_bin()
    if binary is None:
        pytest.skip(
            "Patched DiskANN binary missing. Build it from the "
            "align-diskann-sharded-with-alaya branch in "
            "/md1/huangliang/alaya-dev/Laser/DiskANN, or set DISKANN_ALIGNED_BIN."
        )
    return binary


@pytest.fixture(scope="module")
def synth_dataset_ready() -> None:
    if not SYNTH_BASE_FBIN.exists():
        pytest.skip(f"synth_100k_512d missing at {SYNTH_BASE_FBIN}")


@pytest.fixture(scope="module")
def gist_dataset_ready() -> None:
    if not GIST_BASE_FBIN.exists():
        pytest.skip(f"gist1m missing at {GIST_BASE_FBIN}")


@pytest.mark.usefixtures("synth_dataset_ready")
def test_sharded_generates_baseline(alaya_cli: Path, diskann_aligned_cli: Path) -> None:
    """Regenerate the SHA fixture if absent."""
    if SYNTH_FIXTURE.exists():
        pytest.skip(f"fixture already present: {SYNTH_FIXTURE}")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        alaya_metrics, diskann_metrics = _run_pair(
            alaya_cli, diskann_aligned_cli, SYNTH_BASE_FBIN, SYNTH_BUDGET_GIB, out_dir
        )
        ok, msg = _check_partition_alignment(alaya_metrics, diskann_metrics)
        if not ok:
            pytest.fail(f"Cannot seed fixture — Tier A failed live: {msg}")
        payload = {
            "dataset": "synth_100k_512d",
            "budget_gib": SYNTH_BUDGET_GIB,
            "params": {
                "R": PINNED_R,
                "L": PINNED_L,
                "alpha": PINNED_ALPHA,
                "seed": PINNED_SEED,
                "num_threads": PINNED_THREADS,
            },
            "alaya": alaya_metrics,
            "diskann": diskann_metrics,
        }
        SYNTH_FIXTURE.write_text(json.dumps(payload, indent=2, sort_keys=True))


@pytest.mark.usefixtures("synth_dataset_ready")
def test_sharded_byte_equality_synth(alaya_cli: Path, diskann_aligned_cli: Path) -> None:
    """Tier A primary gate. Compare AlayaLite vs patched DiskANN at matched seeds."""
    if not SYNTH_FIXTURE.exists():
        pytest.skip(f"baseline fixture missing at {SYNTH_FIXTURE}; run test_sharded_generates_baseline first")
    fixture = json.loads(SYNTH_FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        alaya_metrics, diskann_metrics = _run_pair(
            alaya_cli, diskann_aligned_cli, SYNTH_BASE_FBIN, SYNTH_BUDGET_GIB, out_dir
        )
    ok, msg = _check_partition_alignment(alaya_metrics, diskann_metrics)
    assert ok, f"Tier A live disagreement: {msg}"
    expected_alaya_medoids = fixture["alaya"]["medoids_sha256"]
    assert alaya_metrics["medoids_sha256"] == expected_alaya_medoids, (
        f"AlayaLite _medoids.bin drifted from committed fixture {SYNTH_FIXTURE}.\n"
        f"  expected: {expected_alaya_medoids}\n"
        f"  observed: {alaya_metrics['medoids_sha256']}\n"
        f"If intentional, regenerate via "
        f"`rm {SYNTH_FIXTURE} && pytest -k test_sharded_generates_baseline`."
    )


@pytest.mark.extended
@pytest.mark.usefixtures("gist_dataset_ready")
def test_sharded_byte_equality_gist1m(alaya_cli: Path, diskann_aligned_cli: Path) -> None:
    """Extended-marker GIST-1M scenario; manual run before each archive."""
    if not GIST_FIXTURE.exists():
        pytest.skip(
            f"GIST fixture missing at {GIST_FIXTURE}; seed it manually with "
            f"`pytest -m extended -k generates_baseline_gist1m`"
        )
    fixture = json.loads(GIST_FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        alaya_metrics, diskann_metrics = _run_pair(
            alaya_cli, diskann_aligned_cli, GIST_BASE_FBIN, GIST_BUDGET_GIB, out_dir
        )
    ok, msg = _check_partition_alignment(alaya_metrics, diskann_metrics)
    assert ok, f"Tier A GIST-1M live disagreement: {msg}"
    expected_alaya_medoids = fixture["alaya"]["medoids_sha256"]
    assert alaya_metrics["medoids_sha256"] == expected_alaya_medoids, (
        f"AlayaLite GIST-1M _medoids.bin drifted from committed fixture {GIST_FIXTURE}."
    )


@pytest.mark.extended
@pytest.mark.usefixtures("gist_dataset_ready")
def test_sharded_generates_baseline_gist1m(alaya_cli: Path, diskann_aligned_cli: Path) -> None:
    """Manual GIST-1M fixture seeder (extended-marker, not part of CI)."""
    if GIST_FIXTURE.exists():
        pytest.skip(f"GIST fixture already present: {GIST_FIXTURE}")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        alaya_metrics, diskann_metrics = _run_pair(
            alaya_cli, diskann_aligned_cli, GIST_BASE_FBIN, GIST_BUDGET_GIB, out_dir
        )
        ok, msg = _check_partition_alignment(alaya_metrics, diskann_metrics)
        if not ok:
            pytest.fail(f"Cannot seed GIST fixture — Tier A failed live: {msg}")
        payload = {
            "dataset": "gist1m",
            "budget_gib": GIST_BUDGET_GIB,
            "params": {
                "R": PINNED_R,
                "L": PINNED_L,
                "alpha": PINNED_ALPHA,
                "seed": PINNED_SEED,
                "num_threads": PINNED_THREADS,
            },
            "alaya": alaya_metrics,
            "diskann": diskann_metrics,
        }
        GIST_FIXTURE.write_text(json.dumps(payload, indent=2, sort_keys=True))
