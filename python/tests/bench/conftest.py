# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Bench-test collection guards.

Two invariants every bench test depends on, hoisted here so the test
modules stay focused on the assertion body:

1. ``DiskCollection`` v1 is POSIX-only — skip the whole tree on Windows.
2. ``alayalite._alayalitepy`` must be importable (built C++ extension).
"""

from __future__ import annotations

import sys

import pytest

collect_ignore_glob: list[str] = []
if sys.platform == "win32":
    # Skip the entire bench subtree on Windows rather than emitting one
    # ``pytestmark`` skip per module — DiskCollection v1 is POSIX-only.
    collect_ignore_glob.append("*.py")
else:
    pytest.importorskip(
        "alayalite._alayalitepy",
        reason="bench tests require built alayalite extension",
    )
