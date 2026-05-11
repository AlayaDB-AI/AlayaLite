# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for Index.from_prefix loading and search."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _laser_support import DISK_LASER_SUPPORTED  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DISK_LASER_SUPPORTED,
        reason="disk_laser is not supported on this build/platform",
    ),
]


def _vectors(n: int = 512, dim: int = 128, seed: int = 41) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def test_from_prefix_cross_process_and_missing_path(tmp_path: Path) -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    vectors = _vectors()
    prefix = tmp_path / "x"
    idx = laser.Index.fit(
        vectors,
        output_dir=tmp_path,
        name="x",
        build_params=laser.BuildParams(main_dim=128, R=64, disable_medoid=True),
        num_threads=1,
        seed=42,
        auto_load=False,
    )
    with pytest.raises(RuntimeError, match="not loaded"):
        idx.search(vectors[0], 10)

    query_path = tmp_path / "query.npy"
    np.save(query_path, vectors[3])
    code = f"""
import json
import numpy as np
from alayalite import laser
q = np.load(r\"{query_path}\")
idx = laser.Index.from_prefix(r\"{prefix}\", dram_budget_gb=1.0)
print("JSON:" + json.dumps(idx.search(q, 10).tolist()))
"""
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    payload_lines = [line for line in out.splitlines() if line.startswith("JSON:")]
    assert payload_lines, f"missing JSON payload in subprocess output: {out!r}"
    subprocess_hits = json.loads(payload_lines[-1][len("JSON:") :])

    idx2 = laser.Index.from_prefix(prefix, dram_budget_gb=1.0)
    local_hits = idx2.search(vectors[3], 10).tolist()
    assert subprocess_hits == local_hits

    with pytest.raises(FileNotFoundError, match="_R\\*_MD\\*\\.index"):
        laser.Index.from_prefix(tmp_path / "no_such_prefix")
