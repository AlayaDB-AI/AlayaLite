# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bench-test collection guards.

Two invariants every bench test depends on, hoisted here so the test
modules stay focused on the assertion body:

1. ``alayalite._alayalitepy`` must be importable (built C++ extension).
2. ``DiskCollection`` v1 is POSIX-only — skip the whole tree on Windows.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip(
    "alayalite._alayalitepy",
    reason="bench tests require built alayalite extension",
)

collect_ignore_glob: list[str] = []
if sys.platform == "win32":
    # Skip the entire bench subtree on Windows rather than emitting one
    # ``pytestmark`` skip per module — DiskCollection v1 is POSIX-only.
    collect_ignore_glob.append("*.py")
