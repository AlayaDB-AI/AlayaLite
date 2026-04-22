"""Gate G1 second half — CLI and Python binding produce byte-equal output.

Runs both entry points against the same deterministic dataset with
identical parameters and asserts their `.index` outputs are byte-for-byte
identical. A divergence here indicates a default drift between the CLI's
argv parser and the Python binding's pybind arg defaults, or a regression
in the shared `build_dispatch.hpp`.

Dataset presence:
  Requires `/md1/huangliang/alaya-dev/data/synth_100k_512d/base.fbin`.
  Without it, the test skips.

Build presence:
  Requires the CLI binary under `build/tools/build_vamana_index/`. The
  Python binding is imported directly; it must be installed into the
  active venv (via `uv sync --group laser` or similar).
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

# Canonical args pinned by design D8 / Gate G1. `num_threads=1` eliminates
# OpenMP schedule nondeterminism; `build_dram_budget=1.0` is large enough
# for 100k × 512d × float32 (~0.22 GiB) to force the single-shard path.
CANONICAL = {
    "R": 64,
    "L": 100,
    "alpha": 1.2,
    "seed": 1234,
    "num_threads": 1,
    "dram_budget_gb": 1.0,
}


def _find_cli_binary() -> Path | None:
    candidates = [
        REPO_ROOT / "build" / "tools" / "build_vamana_index" / "build_vamana_index",
        REPO_ROOT / "build-release" / "tools" / "build_vamana_index" / "build_vamana_index",
    ]
    for p in candidates:
        if p.exists() and os.access(p, os.X_OK):
            return p
    return None


@pytest.fixture(scope="module")
def cli_binary() -> Path:
    binary = _find_cli_binary()
    if binary is None:
        pytest.skip(
            "build_vamana_index binary not found; run "
            "`cmake --build build --target build_vamana_index` first"
        )
    return binary


@pytest.fixture(scope="module")
def vamana_module():
    """Import the vamana binding; skip if unavailable (e.g., wheel not built)."""
    try:
        from alayalite import vamana
    except ImportError as e:
        pytest.skip(f"alayalite.vamana not importable ({e}); run `uv sync --group laser`")
    return vamana


@pytest.fixture(scope="module")
def dataset_ready() -> None:
    if not SYNTH_BASE_FBIN.exists():
        pytest.skip(
            f"synth_100k_512d dataset missing at {SYNTH_BASE_FBIN}; "
            f"generate via scripts/gen_synth_100k_512d.py"
        )


@pytest.mark.usefixtures("dataset_ready")
def test_cli_vs_python_binding_byte_equality(
    cli_binary: Path, vamana_module
) -> None:
    """Run CLI and Python binding with identical args, then `filecmp.cmp`.

    Runtime on gpu04: ~6 min per build (2 builds → ~12 min total). Marked
    as the slow Gate G1 test; acceptable for pre-merge CI.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cli_out = Path(tmpdir) / "cli.index"
        py_out = Path(tmpdir) / "python.index"

        # CLI invocation — uses short flags (-R/-L/-T) + long flags
        # matching the CLI's argv parser; field names in CANONICAL use the
        # binding's Python kwarg names.
        subprocess.run(
            [
                str(cli_binary),
                "--data_path", str(SYNTH_BASE_FBIN),
                "--index_path_prefix", str(cli_out),
                "-R", str(CANONICAL["R"]),
                "-L", str(CANONICAL["L"]),
                "--alpha", str(CANONICAL["alpha"]),
                "--seed", str(CANONICAL["seed"]),
                "-T", str(CANONICAL["num_threads"]),
                "--build_dram_budget", str(CANONICAL["dram_budget_gb"]),
            ],
            check=True,
        )

        # Python binding invocation — same shared build_dispatch library,
        # so the output must be byte-identical.
        vamana_module.build_index(
            data_path=str(SYNTH_BASE_FBIN),
            output_path=str(py_out),
            R=CANONICAL["R"],
            L=CANONICAL["L"],
            alpha=CANONICAL["alpha"],
            seed=CANONICAL["seed"],
            num_threads=CANONICAL["num_threads"],
            dram_budget_gb=CANONICAL["dram_budget_gb"],
        )

        assert filecmp.cmp(cli_out, py_out, shallow=False), (
            f"CLI output differs from Python binding output.\n"
            f"  cli: {cli_out} ({cli_out.stat().st_size} bytes)\n"
            f"  py : {py_out} ({py_out.stat().st_size} bytes)\n"
            f"Check kDefaultVamanaBuildParams — defaults may have drifted."
        )
