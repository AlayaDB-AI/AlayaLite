# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit coverage for LASER runtime SIMD dispatch exposed through Python."""

from __future__ import annotations

import pytest
from _laser_support import DISK_LASER_SUPPORTED, LASER_SIMD  # noqa: E402


@pytest.mark.skipif(not DISK_LASER_SUPPORTED, reason="disk_laser is not supported on this build/platform")
def test_laser_selected_simd_allows_avx2_fallback() -> None:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    selected = laser.selected_simd()
    print(f"laser_simd={selected}")
    assert selected in {"avx512", "avx2"}
    assert selected == LASER_SIMD
