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

"""Smoke test for `python -m alayalite.bench.laser_compare`.

Implements OpenSpec change `python-laser-native-equivalence-benchmark`
Section 8 (smoke test). Verifies the harness end-to-end on the small
deterministic fixture from `python/tests/fixtures/laser/builder.py`,
checks the comparison block invariants, and asserts the source-dir
no-mutation invariant.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from _laser_support import DISK_LASER_SUPPORTED  # type: ignore[import-not-found]

pytestmark = pytest.mark.skipif(
    not DISK_LASER_SUPPORTED,
    reason="disk_laser not available on this build",
)


def _build_fixture(target: Path) -> Path:
    """Build the small LASER fixture and return its directory."""
    # Lazy: builder needs `alayalite.laser` / `alayalite.vamana`, which
    # only exist on builds with `ALAYA_ENABLE_LASER=ON`.
    from fixtures.laser.builder import build_small_laser_artifacts  # pylint: disable=import-outside-toplevel

    src = target / "src"
    build_small_laser_artifacts(src, n=256, dim=128)
    return src


def _run_compare_cli(src: Path, out_root: Path) -> Path:
    """Invoke the compare CLI as a subprocess and return its run dir."""
    args = [
        sys.executable,
        "-m",
        "alayalite.bench.laser_compare",
        "--laser-src-dir",
        str(src),
        "--vectors",
        str(src / "dsqg_seg_00000001_pca_base.fbin"),
        "--n",
        "256",
        "--dim",
        "128",
        "--queries",
        "8",
        "--warmup",
        "2",
        "--k",
        "10",
        "--num-threads",
        "1",
        "--out",
        str(out_root),
    ]
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, f"laser_compare CLI exited {completed.returncode}: stderr={completed.stderr!r}"
    run_dirs = sorted(p for p in out_root.iterdir() if p.is_dir())
    assert len(run_dirs) == 1, f"expected one run dir under {out_root}, got {[p.name for p in run_dirs]}"
    return run_dirs[0]


def _snapshot_dir(root: Path) -> dict[str, tuple[int, float, str]]:
    """Snapshot every regular file under ``root`` as ``(size, mtime_ns, sha256)``.

    sha256 anchors detection even if a clobber preserves mtime/size.
    """
    snapshot: dict[str, tuple[int, float, str]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        st = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        snapshot[str(path.relative_to(root))] = (st.st_size, st.st_mtime_ns, digest.hexdigest())
    return snapshot


def test_smoke_paired_run(tmp_path: Path) -> None:
    src = _build_fixture(tmp_path / "fixture")
    out_root = tmp_path / "out"
    run_dir = _run_compare_cli(src, out_root)

    summary_json = run_dir / "summary.json"
    assert summary_json.is_file(), f"missing {summary_json}"
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    comparison = summary["comparison"]
    qps_ratio = comparison["qps_ratio"]
    recall_delta = comparison["recall_delta"]
    # qps_ratio band is unstable at this scale: each path's timed loop is
    # tens of microseconds, so CI jitter + cache asymmetry dominate. Parity
    # is verified by the macro benchmark; smoke only asserts well-formed.
    assert qps_ratio is not None and math.isfinite(qps_ratio) and qps_ratio > 0, (
        f"qps_ratio {qps_ratio!r} is not finite-positive"
    )
    assert recall_delta is not None and -0.05 <= recall_delta <= 0.05, (
        f"recall_delta {recall_delta!r} outside [-0.05, 0.05]"
    )

    raws_by_engine = {raw["engine"]: raw["results"] for raw in summary["raws"]}
    for engine in ("native_laser", "disk_laser"):
        results = raws_by_engine[engine]
        qps = results["qps"]
        assert qps > 0, f"{engine} qps not > 0: {qps!r}"
        assert results["latency_us"]["p50"] > 0, f"{engine} latency_us.p50 not > 0"

    raws = summary["raws"]
    n_sha = raws[0]["provenance"]["dataset_sha256_prefix"]
    d_sha = raws[1]["provenance"]["dataset_sha256_prefix"]
    assert n_sha == d_sha, f"dataset_sha256_prefix divergence: native={n_sha!r} disk={d_sha!r}"

    assert (run_dir / "summary.md").is_file()
    raw_dir = run_dir / "raw"
    assert raw_dir.is_dir()
    assert (raw_dir / "native_laser_laser_files_L2.json").is_file()
    assert (raw_dir / "disk_laser_laser_files_L2.json").is_file()


def test_source_dir_unchanged(tmp_path: Path) -> None:
    src = _build_fixture(tmp_path / "fixture")
    before = _snapshot_dir(src)
    _run_compare_cli(src, tmp_path / "out")
    after = _snapshot_dir(src)
    assert before == after, (
        "files under --laser-src-dir were modified during the run; expected "
        "byte-equal (size, mtime, sha256) snapshot before and after"
    )
