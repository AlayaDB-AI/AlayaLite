# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""End-to-end demo of alayalite.DiskCollection(index_type="disk_laser").

Builds a small synthetic LASER fixture in a temp directory using the
upstream `alayalite.laser` + `alayalite.vamana` Python pipeline, imports
it as a single segment, runs a search, and prints the top-k labels.

On unsupported builds (Linux+OFF / macOS / Windows) the example
gracefully prints "skipped: disk_laser not available on this build" and
exits ``0`` so the script is safe to run across the wheel matrix.

NOTE: the wheel must be rebuilt for the target CPU architecture; the
LASER kernel is built with ``-march=native``. A wheel built on a host
with AVX-512 will not load on a CPU that lacks AVX-512.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from alayalite import DiskCollection, MetricType

# Reuse the test-suite fixture builder and the runtime LASER probe via the
# documented sys.path trick (tasks.md 9.1 explicitly allows this for the
# example). This avoids drift from inlining the builder, and the probe
# guarantees `build_small_laser_artifacts` is never invoked on a build
# where its lazy `from alayalite import laser` would raise `ImportError`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python" / "tests"))
# pylint: disable=wrong-import-position
from _laser_support import DISK_LASER_SUPPORTED  # noqa: E402
from fixtures.laser.builder import build_small_laser_artifacts  # noqa: E402


def main() -> int:
    if not DISK_LASER_SUPPORTED:
        print("[disk_laser_demo] skipped: disk_laser not available on this build")
        return 0
    try:
        n = 256
        dim = 128
        with tempfile.TemporaryDirectory(prefix="alayalite_disk_laser_demo_") as tmp:
            tmp_root = Path(tmp)
            src_dir = tmp_root / "src"
            coll_dir = tmp_root / "coll"

            print("[disk_laser_demo] building fixture in", src_dir)
            _, _, labels = build_small_laser_artifacts(src_dir, n=n, dim=dim)

            col = DiskCollection(
                path=str(coll_dir),
                dim=dim,
                metric=MetricType.L2,
                index_type="disk_laser",
            )
            col.import_laser_segment(str(src_dir), labels)
            print(f"[disk_laser_demo] imported segment: size={col.size()}, dim={col.dim()}")

            rng = np.random.default_rng(7)
            query = rng.standard_normal(dim).astype(np.float32)
            hits = col.search(query, k=5, ef=128, beam_width=4)
            print("[disk_laser_demo] top-5 hits (label, distance):")
            for label, distance in hits:
                tag = "nan" if math.isnan(distance) else f"{distance:.4f}"
                print(f"  ({label}, {tag})")

            del col
            reopened = DiskCollection.open(str(coll_dir))
            hits_again = reopened.search(query, k=5, ef=128, beam_width=4)
            assert [label for label, _ in hits] == [label for label, _ in hits_again], (
                "labels must match across reopen (LASER distances are NaN so tuple equality fails)"
            )
            print("[disk_laser_demo] OK")
            return 0
    except (ValueError, RuntimeError) as exc:
        msg = str(exc)
        if "disk_laser" in msg and "not implemented in v1" in msg:
            print("[disk_laser_demo] skipped: disk_laser not available on this build")
            return 0
        raise


if __name__ == "__main__":
    raise SystemExit(main())
