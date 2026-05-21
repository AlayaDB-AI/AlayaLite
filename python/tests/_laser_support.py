# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Runtime probe for `disk_laser` engine support.

Test-only module (NOT part of the public `alayalite` package). The probe
attempts to construct a tiny `DiskCollection(... index_type="disk_laser")`
in a `tempfile.mkdtemp()` directory and caches the result in
`DISK_LASER_SUPPORTED` and `LASER_SIMD`. Tests gate with
`pytest.mark.skipif(not DISK_LASER_SUPPORTED, ...)`.

Per design D6: the probe asks the real C++ gate exactly once (via
`engine_supported_v1`'s effective behaviour at the binding boundary), so
a build / platform matrix change is picked up automatically without
hard-coded `platform.system()` sniffing.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from alayalite import DiskCollection, MetricType


def _probe() -> tuple[bool, Optional[str]]:
    """Attempt a tiny disk_laser constructor; return support status and selected SIMD backend."""
    tmp = Path(tempfile.mkdtemp(prefix="alayalite_laser_probe_"))
    target = tmp / "probe"
    try:
        DiskCollection(
            path=str(target),
            dim=128,
            metric=MetricType.L2,
            index_type="disk_laser",
        )
    except (ValueError, RuntimeError) as exc:
        msg = str(exc)
        if "disk_laser" in msg and "not implemented in v1" in msg:
            return False, None
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    from alayalite import laser  # pylint: disable=import-outside-toplevel

    try:
        laser_simd = laser.selected_simd()
    except RuntimeError as exc:
        if "LASER requires AVX2+FMA" in str(exc):
            return False, None
        raise
    if laser_simd not in {"avx512", "avx2"}:
        raise RuntimeError(f"unexpected LASER SIMD backend: {laser_simd!r}")
    return True, laser_simd


DISK_LASER_SUPPORTED, LASER_SIMD = _probe()
"""Cached at import time so tests pay the probe cost exactly once."""
