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

"""Tests for the alayalite.DiskCollection disk_vamana Python binding."""

import concurrent.futures
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from _laser_support import DISK_LASER_SUPPORTED
from alayalite import DiskCollection, MetricType

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="DiskCollection v1 is POSIX-only")


def _rand_vectors(n, dim, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _ids(n, base=1000, step=1):
    return (base + step * np.arange(n, dtype=np.uint64)).astype(np.uint64)


def _read_manifest(path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        out[key] = value
    return out


def _pythonpath_env(**overrides):
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    pythonpath = str(repo_root / "python" / "src")
    if existing:
        pythonpath = pythonpath + os.pathsep + existing
    env["PYTHONPATH"] = pythonpath
    env.update(overrides)
    return env


def _run_python(code, **env_overrides):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
        env=_pythonpath_env(**env_overrides),
    )


def _build_vamana(path, vectors, ids, **kwargs):
    col = DiskCollection(
        path=str(path),
        dim=vectors.shape[1],
        metric=MetricType.L2,
        index_type="disk_vamana",
        vamana_R=kwargs.get("vamana_R", 16),
        vamana_L=kwargs.get("vamana_L", 64),
        vamana_alpha=kwargs.get("vamana_alpha", 1.1),
        vamana_seed=kwargs.get("vamana_seed", 42),
        vamana_num_threads=kwargs.get("vamana_num_threads", 1),
    )
    col.add(vectors, ids)
    col.flush()
    return col


def test_disk_vamana_basic(tmp_path):
    dim = 16
    vectors = _rand_vectors(256, dim, seed=1)
    ids = _ids(256, base=5000)
    col = _build_vamana(tmp_path / "vamana", vectors, ids)

    assert col.size() == 256
    assert col.dim() == dim

    hits = col.search(vectors[0], k=10, ef=64)
    assert isinstance(hits, list)
    assert len(hits) == 10
    assert all(isinstance(label, int) and isinstance(distance, float) for label, distance in hits)
    assert all(label in set(map(int, ids)) for label, _ in hits)

    manifest = _read_manifest(tmp_path / "vamana" / "segments" / "seg_00000001" / "manifest.txt")
    assert manifest["index_type"] == "disk_vamana"
    assert manifest["x_R"] == "16"
    assert manifest["x_L"] == "64"
    assert float(manifest["x_alpha"]) == pytest.approx(1.1, abs=1e-6)
    assert manifest["x_seed"] == "42"


def test_disk_vamana_reopen(tmp_path):
    dim = 16
    vectors = _rand_vectors(256, dim, seed=2)
    ids = _ids(256, base=7000)
    path = tmp_path / "vamana"
    col = _build_vamana(path, vectors, ids)

    query = _rand_vectors(1, dim, seed=99)[0]
    before = col.search(query, k=10, ef=64)
    del col

    reopened = DiskCollection.open(str(path))
    after = reopened.search(query, k=10, ef=64)
    assert after == before


def test_disk_vamana_external_labels(tmp_path):
    dim = 16
    vectors = _rand_vectors(192, dim, seed=3)
    ids = _ids(192, base=1000, step=17)
    col = _build_vamana(tmp_path / "vamana", vectors, ids)

    hits = col.search(vectors[7], k=10, ef=64)
    supplied = set(map(int, ids))
    assert hits
    assert all(label in supplied for label, _ in hits)
    assert all(label >= 1000 for label, _ in hits)


def test_disk_vamana_recall_against_diskflat_l2(tmp_path):
    dim = 16
    vectors = _rand_vectors(320, dim, seed=4)
    ids = _ids(320, base=9000)
    queries = _rand_vectors(20, dim, seed=5)

    flat = DiskCollection(
        path=str(tmp_path / "flat"),
        dim=dim,
        metric=MetricType.L2,
        index_type="disk_flat",
    )
    flat.add(vectors, ids)
    flat.flush()
    vamana = _build_vamana(tmp_path / "vamana", vectors, ids, vamana_R=32, vamana_L=96)

    recalls = []
    for query in queries:
        exact = {label for label, _ in flat.search(query, k=10)}
        approx = {label for label, _ in vamana.search(query, k=10, ef=128)}
        recalls.append(len(exact & approx) / 10.0)
    assert float(np.mean(recalls)) >= 0.7


def test_disk_vamana_duplicate_label_throws(tmp_path):
    dim = 8
    vectors = _rand_vectors(64, dim, seed=6)
    ids = _ids(64)
    ids[17] = ids[3]
    col = DiskCollection(
        path=str(tmp_path / "vamana"),
        dim=dim,
        metric=MetricType.L2,
        index_type="disk_vamana",
        vamana_R=16,
        vamana_L=32,
    )
    col.add(vectors, ids)

    with pytest.raises(Exception) as exc_info:
        col.flush()
    assert "duplicate" in str(exc_info.value).lower()
    assert not (tmp_path / "vamana" / "segments" / "seg_00000001").exists()


@pytest.mark.parametrize(
    "kwargs, token",
    [
        ({"vamana_R": 0}, "vamana_R"),
        ({"vamana_R": -1}, "vamana_R"),
        ({"vamana_R": 2**32}, "vamana_R"),
        ({"vamana_L": 0}, "vamana_L"),
        ({"vamana_L": -1}, "vamana_L"),
        ({"vamana_L": 2**32}, "vamana_L"),
        ({"vamana_R": 32, "vamana_L": 16}, "vamana_L"),
        ({"vamana_alpha": 0.99}, "vamana_alpha"),
        ({"vamana_alpha": float("nan")}, "vamana_alpha"),
        ({"vamana_alpha": float("inf")}, "vamana_alpha"),
        ({"vamana_alpha": 1e39}, "vamana_alpha"),
        ({"vamana_seed": -1}, "vamana_seed"),
        ({"vamana_seed": 2**32}, "vamana_seed"),
        ({"vamana_num_threads": -1}, "vamana_num_threads"),
        ({"vamana_num_threads": 2**32}, "vamana_num_threads"),
    ],
)
def test_disk_vamana_invalid_params_throw(tmp_path, kwargs, token):
    path = tmp_path / f"bad_{token}_{len(str(kwargs))}"
    with pytest.raises(ValueError) as exc_info:
        DiskCollection(
            path=str(path),
            dim=8,
            metric=MetricType.L2,
            index_type="disk_vamana",
            **kwargs,
        )
    assert token in str(exc_info.value)
    assert not path.exists()


def test_disk_vamana_honors_omp_num_threads_default(tmp_path):
    result = _run_python(
        f"""
        from pathlib import Path
        from alayalite import DiskCollection, MetricType

        path = Path({str(tmp_path / "vamana")!r})
        DiskCollection(
            path=str(path),
            dim=8,
            metric=MetricType.L2,
            index_type="disk_vamana",
        )
        print((path / "collection_manifest.txt").read_text())
        """,
        OMP_NUM_THREADS="4",
    )
    assert result.returncode == 0, result.stderr
    assert "x_vamana_num_threads=4" in result.stdout


def test_disk_vamana_explicit_num_threads_overrides_omp_default(tmp_path):
    result = _run_python(
        f"""
        from pathlib import Path
        from alayalite import DiskCollection, MetricType

        path = Path({str(tmp_path / "vamana")!r})
        DiskCollection(
            path=str(path),
            dim=8,
            metric=MetricType.L2,
            index_type="disk_vamana",
            vamana_num_threads=2,
        )
        print((path / "collection_manifest.txt").read_text())
        """,
        OMP_NUM_THREADS="4",
    )
    assert result.returncode == 0, result.stderr
    assert "x_vamana_num_threads=2" in result.stdout


def test_disk_vamana_dtype_errors(tmp_path):
    dim = 8
    path = tmp_path / "vamana"
    col = DiskCollection(
        path=str(path),
        dim=dim,
        metric=MetricType.L2,
        index_type="disk_vamana",
        vamana_R=16,
        vamana_L=32,
    )

    with pytest.raises((TypeError, ValueError)) as exc_info:
        col.add(np.zeros((4, dim), dtype=np.float64), _ids(4))
    assert "float32" in str(exc_info.value)

    with pytest.raises((TypeError, ValueError)) as exc_info:
        col.add(np.zeros((4, dim), dtype=np.float32), np.arange(4, dtype=np.int32))
    assert "uint64" in str(exc_info.value)

    with pytest.raises((TypeError, ValueError)) as exc_info:
        col.add(np.zeros((4, dim * 2), dtype=np.float32)[:, ::2], _ids(4))
    assert "contiguous" in str(exc_info.value).lower()

    vectors = _rand_vectors(64, dim, seed=7)
    ids = _ids(64)
    col.add(vectors, ids)
    col.flush()

    with pytest.raises(ValueError):
        col.search(np.zeros(dim + 1, dtype=np.float32), k=10, ef=64)

    with pytest.raises((TypeError, ValueError)) as exc_info:
        col.search(np.zeros(dim * 2, dtype=np.float32)[::2], k=10, ef=64)
    assert "contiguous" in str(exc_info.value).lower()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_disk_vamana_search_rejects_non_finite_query(tmp_path, bad):
    dim = 8
    vectors = _rand_vectors(64, dim, seed=8)
    ids = _ids(64)
    col = _build_vamana(tmp_path / "vamana", vectors, ids)
    query = vectors[0].copy()
    query[3] = bad

    with pytest.raises((TypeError, ValueError)) as exc_info:
        col.search(query, k=10, ef=64)
    msg = str(exc_info.value).lower()
    assert "query" in msg
    assert "nan" in msg or "inf" in msg or "finite" in msg


def test_disk_vamana_top_k_exceeds_count_caps_even_when_ef_is_smaller(tmp_path):
    dim = 8
    vectors = _rand_vectors(32, dim, seed=9)
    ids = _ids(32)
    col = _build_vamana(tmp_path / "vamana", vectors, ids)

    hits = col.search(vectors[0], k=200, ef=10)
    assert len(hits) == 32


def test_disk_vamana_huge_ef_clamps_to_segment_count(tmp_path):
    result = _run_python(
        f"""
        import pathlib
        import numpy as np
        from alayalite import DiskCollection, MetricType

        rng = np.random.default_rng(123)
        vectors = rng.standard_normal((32, 8)).astype(np.float32)
        ids = np.arange(1000, 1032, dtype=np.uint64)
        path = pathlib.Path({str(tmp_path / "vamana_huge_ef")!r})
        col = DiskCollection(
            path=str(path),
            dim=8,
            metric=MetricType.L2,
            index_type="disk_vamana",
            vamana_R=16,
            vamana_L=64,
            vamana_num_threads=1,
        )
        col.add(vectors, ids)
        col.flush()
        hits = col.search(vectors[0], k=10, ef=2**31 - 1)
        assert 0 < len(hits) <= 10
        print("ok", len(hits))
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "ok" in result.stdout


def test_disk_vamana_singleton_flush_throws_before_publish(tmp_path):
    path = tmp_path / "vamana"
    col = DiskCollection(
        path=str(path),
        dim=4,
        metric=MetricType.L2,
        index_type="disk_vamana",
        vamana_R=1,
        vamana_L=1,
        vamana_num_threads=1,
    )
    col.add(np.ones((1, 4), dtype=np.float32), np.array([123], dtype=np.uint64))

    with pytest.raises(Exception) as exc_info:
        col.flush()
    assert "row" in str(exc_info.value).lower() or "count" in str(exc_info.value).lower()
    assert not (path / "segments" / "seg_00000001").exists()


@pytest.mark.parametrize(
    "body, token",
    [
        (
            "version=1\ndim=4\nmetric=IP\nindex_type=disk_vamana\nnext_segment_id=1\n",
            "metric",
        ),
        (
            "version=1\ndim=4\nmetric=L2\nindex_type=disk_vamana\nnext_segment_id=1\nx_vamana_R=0\n",
            "x_vamana_R",
        ),
        (
            "version=1\ndim=4\nmetric=L2\nindex_type=disk_vamana\nnext_segment_id=1\nx_vamana_L=0\n",
            "x_vamana_L",
        ),
        (
            "version=1\ndim=4\nmetric=L2\nindex_type=disk_vamana\nnext_segment_id=1\nx_vamana_R=64\nx_vamana_L=32\n",
            "x_vamana_L",
        ),
        (
            "version=1\ndim=4\nmetric=L2\nindex_type=disk_vamana\nnext_segment_id=1\nx_vamana_alpha=inf\n",
            "x_vamana_alpha",
        ),
        (
            "version=1\ndim=4\nmetric=L2\nindex_type=disk_vamana\nnext_segment_id=1\nx_vamana_alpha=1.2junk\n",
            "x_vamana_alpha",
        ),
        (
            "version=1\ndim=4\nmetric=L2\nindex_type=disk_vamana\nnext_segment_id=1\nx_vamana_seed=4294967296\n",
            "x_vamana_seed",
        ),
        (
            "version=1\ndim=4\nmetric=L2\nindex_type=disk_vamana\nnext_segment_id=1\nx_vamana_num_threads=4294967296\n",
            "x_vamana_num_threads",
        ),
    ],
)
def test_disk_vamana_open_rejects_invalid_persisted_config(tmp_path, body, token):
    path = tmp_path / "vamana"
    (path / "segments").mkdir(parents=True)
    (path / "collection_manifest.txt").write_text(body, encoding="utf-8")

    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        DiskCollection.open(str(path))
    assert token in str(exc_info.value)
    assert not (path / "segments" / "seg_00000001").exists()


@pytest.mark.parametrize("index_type", ["disk_flat", "disk_vamana"])
def test_max_pending_bytes_survives_reopen(tmp_path, index_type):
    path = tmp_path / index_type
    col = DiskCollection(
        path=str(path),
        dim=4,
        metric=MetricType.L2,
        index_type=index_type,
        max_pending_bytes=100,
    )
    del col

    reopened = DiskCollection.open(str(path))
    vectors = np.zeros((3, 4), dtype=np.float32)
    ids = np.arange(3, dtype=np.uint64)
    with pytest.raises(RuntimeError) as exc_info:
        reopened.add(vectors, ids)
    assert "max_pending_bytes" in str(exc_info.value)


def test_concurrent_add_does_not_corrupt_process(tmp_path):
    result = _run_python(
        f"""
        import pathlib
        import threading
        import numpy as np
        from alayalite import DiskCollection, MetricType

        path = pathlib.Path({str(tmp_path / "flat")!r})
        col = DiskCollection(path=str(path), dim=8, metric=MetricType.L2, index_type="disk_flat")
        barrier = threading.Barrier(8)

        def worker(t):
            vectors = np.full((1, 8), t, dtype=np.float32)
            base = np.array([t], dtype=np.uint64)
            barrier.wait()
            for i in range(200):
                col.add(vectors, base + i * 8)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "ok" in result.stdout


def test_concurrent_disk_vamana_add_flush_search_integrity(tmp_path):
    result = _run_python(
        f"""
        import pathlib
        import threading
        import numpy as np
        from alayalite import DiskCollection, MetricType

        path = pathlib.Path({str(tmp_path / "vamana_add")!r})
        col = DiskCollection(
            path=str(path),
            dim=8,
            metric=MetricType.L2,
            index_type="disk_vamana",
            vamana_R=16,
            vamana_L=64,
            vamana_num_threads=1,
        )
        barrier = threading.Barrier(4)

        def worker(t):
            rng = np.random.default_rng(100 + t)
            vectors = rng.standard_normal((16, 8)).astype(np.float32)
            ids = np.arange(t * 1000, t * 1000 + 16, dtype=np.uint64)
            barrier.wait()
            col.add(vectors, ids)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        col.flush()
        assert col.size() == 64
        query = np.zeros(8, dtype=np.float32)
        hits = col.search(query, k=10, ef=64)
        assert len(hits) == 10
        print("ok", col.size(), len(hits))
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "ok" in result.stdout


def test_concurrent_size_dim_during_flush_are_safe(tmp_path):
    result = _run_python(
        f"""
        import pathlib
        import threading
        import numpy as np
        from alayalite import DiskCollection, MetricType

        path = pathlib.Path({str(tmp_path / "flat_size")!r})
        col = DiskCollection(path=str(path), dim=8, metric=MetricType.L2, index_type="disk_flat")
        stop = threading.Event()
        errors = []

        def writer():
            try:
                for batch in range(100):
                    vectors = np.full((4, 8), batch, dtype=np.float32)
                    ids = np.arange(batch * 4, batch * 4 + 4, dtype=np.uint64)
                    col.add(vectors, ids)
                    col.flush()
            except Exception as exc:
                errors.append(repr(exc))
            finally:
                stop.set()

        def reader():
            try:
                while not stop.is_set():
                    if col.dim() != 8:
                        errors.append("bad dim")
                    if col.size() < 0:
                        errors.append("bad size")
            except Exception as exc:
                errors.append(repr(exc))

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=writer))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError(errors)
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "ok" in result.stdout


def test_concurrent_disk_vamana_search_results_are_stable(tmp_path):
    dim = 16
    vectors = _rand_vectors(256, dim, seed=10)
    ids = _ids(256)
    col = _build_vamana(tmp_path / "vamana", vectors, ids, vamana_R=16, vamana_L=64)
    query = vectors[0].copy()
    baseline = col.search(query, k=10, ef=64)

    def run_once(_):
        return col.search(query, k=10, ef=64)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_once, range(80)))
    assert all(result == baseline for result in results)


@pytest.mark.parametrize("metric, token", [(MetricType.IP, "IP"), (MetricType.COS, "COS")])
def test_disk_vamana_unsupported_metric_if_applicable(tmp_path, metric, token):
    path = tmp_path / f"vamana_{token.lower()}"
    with pytest.raises(ValueError) as exc_info:
        DiskCollection(
            path=str(path),
            dim=8,
            metric=metric,
            index_type="disk_vamana",
        )
    msg = str(exc_info.value)
    assert token in msg
    assert "supports L2 only" in msg
    assert not path.exists()


@pytest.mark.skipif(
    DISK_LASER_SUPPORTED,
    reason="disk_laser is supported on this build; the rejection contract is "
    "pinned by tests in test_disk_collection_dispatch.py and "
    "test_disk_collection_laser.py::test_disk_laser_unsupported_platform",
)
def test_disk_laser_still_unsupported(tmp_path):
    path = tmp_path / "laser"
    with pytest.raises(ValueError) as exc_info:
        DiskCollection(
            path=str(path),
            dim=8,
            metric=MetricType.L2,
            index_type="disk_laser",
        )
    msg = str(exc_info.value)
    assert "disk_laser" in msg
    assert "not implemented in v1" in msg
    assert not path.exists()


def test_disk_vamana_benchmark_harness_writes_outputs_and_cleans_scratch(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alayalite.bench.disk_collection",
            "--engine",
            "disk_vamana",
            "--dataset",
            "synth",
            "--n",
            "16",
            "--dim",
            "4",
            "--queries",
            "1",
            "--k",
            "3",
            "--ef",
            "8",
            "--warmup",
            "0",
            "--vamana-R",
            "4",
            "--vamana-L",
            "8",
            "--out",
            str(tmp_path),
            "--run-id",
            "vamana",
            "--sweep",
            "off",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_pythonpath_env(),
    )
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / "vamana"
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "raw" / "disk_vamana_synth_L2.json").exists()
    assert not (run_dir / "_scratch").exists()
