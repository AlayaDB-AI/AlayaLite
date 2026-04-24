"""Gate G1 first half — CLI byte output is deterministic and matches the
pinned pre-refactor baseline.

Runs `build_vamana_index` against `synth_100k_512d` with deterministic
parameters (num_threads=1, seed=1234) and compares the output to a fixture
committed (or regenerated locally) at
`tests/vamana/fixtures/synth_100k_512d_baseline.index`. Byte divergence
indicates one of:
  * the CLI dispatch library (`build_dispatch.hpp`) changed build semantics;
  * `VamanaBuilder`'s single-threaded behaviour changed;
  * `save_graph`'s serialization format changed.

Dataset presence:
  Requires the synth_100k_512d .fbin at
  `/md1/huangliang/alaya-dev/data/synth_100k_512d/base.fbin`. Without it,
  the test skips with a clear reason.

Fixture regeneration (when an intentional default change lands):
  rm tests/vamana/fixtures/synth_100k_512d_baseline.index
  pytest tests/vamana/test_cli_byte_equality.py::test_cli_generates_baseline
"""

from __future__ import annotations

import filecmp
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

SYNTH_DATASET = Path("/md1/huangliang/alaya-dev/data/synth_100k_512d")
SYNTH_BASE_FBIN = SYNTH_DATASET / "base.fbin"

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
BASELINE_FIXTURE = FIXTURE_DIR / "synth_100k_512d_baseline.index"

# Canonical args pinned by design D8 / Gate G1.
CANONICAL_ARGS = {
    "R": "64",
    "L": "100",
    "alpha": "1.2",
    "seed": "1234",
    "num_threads": "1",
    "build_dram_budget": "1.0",
}


def _find_cli_binary() -> Path | None:
    """Search common build-output locations for the CLI binary."""
    candidates = [
        REPO_ROOT / "build" / "tools" / "build_vamana_index" / "build_vamana_index",
        REPO_ROOT / "build-release" / "tools" / "build_vamana_index" / "build_vamana_index",
    ]
    for p in candidates:
        if p.exists() and os.access(p, os.X_OK):
            return p
    return None


def _run_cli(binary: Path, output_path: Path) -> None:
    """Run the CLI with canonical Gate G1 args and write to output_path."""
    subprocess.run(
        [
            str(binary),
            "--data_path", str(SYNTH_BASE_FBIN),
            "--index_path_prefix", str(output_path),
            "-R", CANONICAL_ARGS["R"],
            "-L", CANONICAL_ARGS["L"],
            "--alpha", CANONICAL_ARGS["alpha"],
            "--seed", CANONICAL_ARGS["seed"],
            "-T", CANONICAL_ARGS["num_threads"],
            "--build_dram_budget", CANONICAL_ARGS["build_dram_budget"],
        ],
        check=True,
    )


@pytest.fixture(scope="module")
def cli_binary() -> Path:
    binary = _find_cli_binary()
    if binary is None:
        pytest.skip(
            "build_vamana_index binary not found under build/tools/build_vamana_index/; "
            "run `cmake --build build --target build_vamana_index` first"
        )
    return binary


@pytest.fixture(scope="module")
def dataset_ready() -> None:
    if not SYNTH_BASE_FBIN.exists():
        pytest.skip(
            f"synth_100k_512d dataset missing at {SYNTH_BASE_FBIN}; "
            f"generate via scripts/laser_alignment/gen_synth_100k_512d.py"
        )


@pytest.mark.usefixtures("dataset_ready")
def test_cli_generates_baseline(cli_binary: Path) -> None:
    """Regeneration path — run the CLI into the fixture location if missing.

    This test is the "fixture seeder". It runs whenever
    `BASELINE_FIXTURE` is absent so a local dev environment can create a
    baseline on-demand. When the baseline exists, the test skips —
    byte-comparison lives in `test_cli_matches_baseline` below.
    """
    if BASELINE_FIXTURE.exists():
        pytest.skip(f"baseline fixture already exists at {BASELINE_FIXTURE}")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _run_cli(cli_binary, BASELINE_FIXTURE)
    assert BASELINE_FIXTURE.exists()
    assert BASELINE_FIXTURE.stat().st_size > 24  # header is 24 bytes


@pytest.mark.usefixtures("dataset_ready")
def test_cli_matches_baseline(cli_binary: Path) -> None:
    """Gate G1 first half — byte-compare CLI output against pinned baseline."""
    if not BASELINE_FIXTURE.exists():
        pytest.skip(
            f"baseline fixture missing at {BASELINE_FIXTURE}; "
            f"run test_cli_generates_baseline first to seed it"
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        cli_output = Path(tmpdir) / "cli_output.index"
        _run_cli(cli_binary, cli_output)
        assert filecmp.cmp(BASELINE_FIXTURE, cli_output, shallow=False), (
            f"CLI output diverged from baseline fixture.\n"
            f"  baseline: {BASELINE_FIXTURE} ({BASELINE_FIXTURE.stat().st_size} bytes)\n"
            f"  cli_out : {cli_output} ({cli_output.stat().st_size} bytes)\n"
            f"If this change is intentional, regenerate the fixture: "
            f"rm {BASELINE_FIXTURE} && pytest {Path(__file__).name}::test_cli_generates_baseline"
        )
