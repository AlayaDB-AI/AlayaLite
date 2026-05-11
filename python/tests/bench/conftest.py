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
