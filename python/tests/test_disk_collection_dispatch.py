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

"""Python-level tests for the DiskCollection factory dispatch refactor.

Pinned by `disk-segment-searcher-dispatch`:
- `index_type="disk_flat"` keeps working unchanged.
- `index_type="disk_vamana"` / `"disk_laser"` keep raising, with both the
  engine name and the literal "not implemented in v1" present in the error
  message so the Python rejection contract aligns with the C++ factory.
"""

import sys

import numpy as np
import pytest

from alayalite import DiskCollection, MetricType

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="DiskCollection v1 is POSIX-only"
)


def _rand_vectors(n, dim, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _ids(n, base=1000):
    return np.arange(base, base + n, dtype=np.uint64)


def test_constructor_disk_flat_unchanged(tmp_path):
    """Same shape as existing smoke; proves Flat dispatch works after refactor."""
    path = str(tmp_path / "coll")
    col = DiskCollection(
        path=path, dim=128, metric=MetricType.L2, index_type="disk_flat"
    )
    v = _rand_vectors(1000, 128)
    ids = _ids(1000)
    col.add(v, ids)
    col.flush()
    assert col.size() == 1000
    assert col.dim() == 128

    q = _rand_vectors(1, 128, seed=99)[0]
    hits = col.search(q, k=10)
    assert len(hits) == 10
    distances = [d for _, d in hits]
    assert distances == sorted(distances), "hits must be ascending by distance"


@pytest.mark.parametrize("engine", ["disk_vamana", "disk_laser"])
def test_constructor_unsupported_engine_still_rejected(tmp_path, engine):
    """Both engine name AND v1 marker must appear in the rejection message."""
    path = str(tmp_path / f"coll_{engine}")
    with pytest.raises(ValueError) as exc_info:
        DiskCollection(
            path=path, dim=4, metric=MetricType.L2, index_type=engine
        )
    msg = str(exc_info.value)
    assert engine in msg, f"error message must name the engine '{engine}': {msg}"
    assert "not implemented in v1" in msg, (
        f"error message must include the v1 capability marker: {msg}"
    )
    # No partial collection should remain on disk after rejection.
    assert not (tmp_path / f"coll_{engine}").exists()


def test_open_disk_vamana_manifest_rejected_at_python_boundary(tmp_path):
    """C++ can open disk_vamana; Python exposure remains deferred in v1."""
    coll = tmp_path / "coll_vamana"
    (coll / "segments").mkdir(parents=True)
    (coll / "collection_manifest.txt").write_text(
        "\n".join(
            [
                "version=1",
                "dim=4",
                "metric=L2",
                "index_type=disk_vamana",
                "next_segment_id=1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        DiskCollection.open(str(coll))
    msg = str(exc_info.value)
    assert "disk_vamana" in msg
    assert "not implemented in v1" in msg
