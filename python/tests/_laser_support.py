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

"""Runtime probe for `disk_laser` engine support.

Test-only module (NOT part of the public `alayalite` package). The probe
attempts to construct a tiny `DiskCollection(... index_type="disk_laser")`
in a `tempfile.mkdtemp()` directory and caches the result in
`DISK_LASER_SUPPORTED`. Tests gate with
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

from alayalite import DiskCollection, MetricType


def _probe() -> bool:
    """Attempt a tiny disk_laser constructor; return True iff it succeeds."""
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
            return False
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return True


DISK_LASER_SUPPORTED: bool = _probe()
"""Cached at import time so tests pay the probe cost exactly once."""
