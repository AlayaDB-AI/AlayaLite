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
import struct
import tempfile
from pathlib import Path

import numpy as np
from alayalite import DiskCollection, MetricType


def _build_fixture(target_dir: Path, *, n: int = 256, dim: int = 128, R: int = 64, seed: int = 1234):
    """Local fixture builder mirroring `python/tests/fixtures/laser/builder.py`.

    Inlined here so the example does not depend on the test-suite path. Lazy
    imports of ``alayalite.laser`` / ``alayalite.vamana`` happen here so the
    surrounding example file loads cleanly on unsupported builds.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, np.float32(1e-6))

    pca_base = target_dir / "dsqg_seg_00000001_pca_base.fbin"
    with pca_base.open("wb") as f:
        np.array([n, dim], dtype=np.int32).tofile(f)
        vectors.tofile(f)

    # pylint: disable=import-outside-toplevel
    # Intentional lazy import: alayalite.laser / vamana are not loadable on
    # builds without ALAYA_ENABLE_LASER=ON. Keeping the import here lets the
    # example file load cleanly on those builds; the `try` in main() below
    # converts the eventual ValueError into a graceful skip message.
    from alayalite import laser as laser_module
    from alayalite import vamana as vamana_module

    vamana_graph = target_dir / "dsqg_seg_00000001_vamana_graph.index"
    vamana_module.build_index(
        data_path=str(pca_base),
        output_path=str(vamana_graph),
        R=R,
        L=max(R + 8, 100),
        alpha=1.2,
        seed=seed,
        num_threads=1,
        dram_budget_gb=1.0,
    )

    laser_index = laser_module.Index(
        index_type="QG",
        metric="l2",
        num_elements=n,
        main_dimension=dim,
        dimension=dim,
        degree_bound=R,
        rotator_seed=seed,
        rotator_dump_path="",
    )
    laser_index.build_index(
        str(vamana_graph),
        str(target_dir / "dsqg_seg_00000001"),
        EF=200,
        num_thread=1,
    )

    # Sanity: verify the four required artifacts are present.
    prefix = f"dsqg_seg_00000001_R{R}_MD{dim}"
    for suffix in (".index", ".index_rotator", ".index_cache_ids", ".index_cache_nodes"):
        path = target_dir / f"{prefix}{suffix}"
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"LASER artifact missing or empty: {path}")
    with (target_dir / f"{prefix}.index").open("rb") as f:
        head = f.read(8)
    declared_n = struct.unpack("<Q", head)[0]
    if declared_n != n:
        raise RuntimeError(f"LASER index declared count={declared_n} differs from n={n}")

    labels = (1_000_000 + np.arange(n, dtype=np.uint64)).astype(np.uint64)
    return vectors, labels


def main() -> int:
    try:
        n = 256
        dim = 128
        with tempfile.TemporaryDirectory(prefix="alayalite_disk_laser_demo_") as tmp:
            tmp_root = Path(tmp)
            src_dir = tmp_root / "src"
            coll_dir = tmp_root / "coll"

            print("[disk_laser_demo] building fixture in", src_dir)
            _, labels = _build_fixture(src_dir, n=n, dim=dim)

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

            # Reopen and re-search to demonstrate persistence.
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
