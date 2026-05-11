# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the ann-benchmarks AlayaLite disk_laser adapter."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


class _FakeBuildParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_adapter_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "python" / "adapters" / "annbenchmark" / "alayalite" / "module.py"

    ann_pkg = types.ModuleType("ann_benchmarks")
    ann_pkg.__path__ = []
    algo_pkg = types.ModuleType("ann_benchmarks.algorithms")
    algo_pkg.__path__ = []
    base_pkg = types.ModuleType("ann_benchmarks.algorithms.base")
    base_pkg.__path__ = []
    alaya_pkg = types.ModuleType("ann_benchmarks.algorithms.alayalite")
    alaya_pkg.__path__ = []
    base_module = types.ModuleType("ann_benchmarks.algorithms.base.module")

    class BaseANN:
        pass

    base_module.BaseANN = BaseANN

    class _Stub:
        pass

    fake_alayalite = types.ModuleType("alayalite")
    fake_alayalite.Client = _Stub
    fake_alayalite.Index = _Stub
    fake_alayalite.laser = types.SimpleNamespace(Index=_Stub, BuildParams=_FakeBuildParams)

    monkeypatch.setitem(sys.modules, "ann_benchmarks", ann_pkg)
    monkeypatch.setitem(sys.modules, "ann_benchmarks.algorithms", algo_pkg)
    monkeypatch.setitem(sys.modules, "ann_benchmarks.algorithms.base", base_pkg)
    monkeypatch.setitem(sys.modules, "ann_benchmarks.algorithms.base.module", base_module)
    monkeypatch.setitem(sys.modules, "ann_benchmarks.algorithms.alayalite", alaya_pkg)
    monkeypatch.setitem(sys.modules, "alayalite", fake_alayalite)

    spec = importlib.util.spec_from_file_location("ann_benchmarks.algorithms.alayalite.module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _method_param(**overrides):
    params = {
        "index_type": "NONE",
        "quantization_type": "RABITQ",
        "fit_threads": 2,
        "search_threads": 3,
        "R": 64,
        "L": 100,
        "M": 32,
        "alpha": 1.2,
        "seed": 42,
        "beam_width": 16,
        "work_root": "/tmp/annbenchmark-disk-laser-test",
    }
    params.update(overrides)
    return params


def test_main_dim_defaults(monkeypatch):
    module = _load_adapter_module(monkeypatch)

    adapter_128 = module.AlayaLiteDiskLaser("l2", 128, _method_param())
    adapter_960 = module.AlayaLiteDiskLaser("l2", 960, _method_param())
    adapter_768 = module.AlayaLiteDiskLaser("l2", 768, _method_param())

    assert adapter_128.main_dim == 128
    assert adapter_960.main_dim == 256
    assert adapter_768.main_dim == 256


def test_main_dim_override_via_method_param(monkeypatch):
    module = _load_adapter_module(monkeypatch)

    adapter = module.AlayaLiteDiskLaser("l2", 960, _method_param(main_dim=512))
    assert adapter.main_dim == 512
    assert adapter.raw_dim == 960

    adapter_legacy = module.AlayaLiteDiskLaser("l2", 960, _method_param(pca_dim=512))
    assert adapter_legacy.main_dim == 512

    with pytest.raises(ValueError, match="power of two"):
        module.AlayaLiteDiskLaser("l2", 960, _method_param(main_dim=300))
    with pytest.raises(ValueError, match=">= 128"):
        module.AlayaLiteDiskLaser("l2", 960, _method_param(main_dim=64))
    with pytest.raises(ValueError, match="<= raw dim"):
        module.AlayaLiteDiskLaser("l2", 960, _method_param(main_dim=2048))


def test_disk_laser_init_rejects_non_l2_metric(monkeypatch):
    module = _load_adapter_module(monkeypatch)
    with pytest.raises(ValueError, match="L2"):
        module.AlayaLiteDiskLaser("cosine", 128, _method_param())


def test_fit_calls_unified_index_fit(monkeypatch, tmp_path):
    module = _load_adapter_module(monkeypatch)
    calls = []

    class FakeIndex:
        @classmethod
        def fit(cls, vectors, **kwargs):
            calls.append(("fit", vectors.shape, kwargs))
            return object()

        @classmethod
        def from_prefix(cls, prefix, dram_budget_gb=1.0):
            raise AssertionError("from_prefix should not run during fit()")

    monkeypatch.setattr(
        module,
        "laser",
        types.SimpleNamespace(Index=FakeIndex, BuildParams=_FakeBuildParams),
        raising=False,
    )
    monkeypatch.setattr(module, "_laser_runtime_supported", lambda: True, raising=False)

    adapter = module.AlayaLiteDiskLaser("euclidean", 256, _method_param(work_root=str(tmp_path)))
    vectors = np.random.default_rng(0).standard_normal((512, 256)).astype(np.float32)
    adapter.fit(vectors)

    assert calls and calls[0][0] == "fit"
    assert calls[0][1] == (512, 256)
    kwargs = calls[0][2]
    assert kwargs["name"] == "dsqg_seg_00000001"
    build_params = kwargs["build_params"]
    assert build_params.metric == "l2"
    assert build_params.main_dim == 256
    assert build_params.R == 64
    assert kwargs["auto_load"] is False
    assert kwargs["skip_existing"] is False
    assert kwargs["num_threads"] == 2
    assert adapter.laser_index is None
    assert adapter._built_prefix is not None  # pylint: disable=protected-access


def test_fit_rejects_external_vamana_path(monkeypatch, tmp_path):
    module = _load_adapter_module(monkeypatch)
    monkeypatch.setattr(module, "_laser_runtime_supported", lambda: True, raising=False)
    adapter = module.AlayaLiteDiskLaser(
        "l2",
        256,
        _method_param(work_root=str(tmp_path), external_vamana_path="/tmp/graph.index"),
    )
    vectors = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    with pytest.raises(ValueError, match="external_vamana_path"):
        adapter.fit(vectors)


def test_runtime_probe_failure_raises(monkeypatch, tmp_path):
    module = _load_adapter_module(monkeypatch)
    monkeypatch.setattr(module, "_laser_runtime_supported", lambda: False, raising=False)
    adapter = module.AlayaLiteDiskLaser("l2", 256, _method_param(work_root=str(tmp_path)))
    vectors = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    with pytest.raises(RuntimeError, match="LASER-enabled build"):
        adapter.fit(vectors)


def test_set_query_arguments_loads_from_prefix_and_sets_params(monkeypatch, tmp_path):
    module = _load_adapter_module(monkeypatch)
    calls = []

    class FakeLoaded:
        def set_params(self, ef_search=200, num_threads=1, beam_width=16):
            calls.append(("set_params", ef_search, num_threads, beam_width))

    class FakeIndex:
        @classmethod
        def fit(cls, vectors, **kwargs):
            calls.append(("fit", vectors.shape, kwargs))
            return object()

        @classmethod
        def from_prefix(cls, prefix, dram_budget_gb=1.0):
            calls.append(("from_prefix", prefix, dram_budget_gb))
            return FakeLoaded()

    monkeypatch.setattr(
        module,
        "laser",
        types.SimpleNamespace(Index=FakeIndex, BuildParams=_FakeBuildParams),
        raising=False,
    )
    monkeypatch.setattr(module, "_laser_runtime_supported", lambda: True, raising=False)

    adapter = module.AlayaLiteDiskLaser("l2", 256, _method_param(work_root=str(tmp_path), search_dram_budget_gb=2.5))
    vectors = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    adapter.fit(vectors)
    adapter.set_query_arguments(77)

    from_prefix = next(entry for entry in calls if entry[0] == "from_prefix")
    assert from_prefix[2] == 2.5
    set_params = next(entry for entry in calls if entry[0] == "set_params")
    assert set_params[1:] == (77, 3, 16)


def test_prepare_query_run_and_batch_query(monkeypatch, tmp_path):
    module = _load_adapter_module(monkeypatch)

    class FakeLoaded:  # pylint: disable=missing-class-docstring
        def __init__(self):
            self.last_params = None

        def set_params(self, ef_search=200, num_threads=1, beam_width=16):
            self.last_params = (ef_search, num_threads, beam_width)

        def search(self, query, k):
            assert query.shape == (256,)
            return np.array([9, 7, 3], dtype=np.uint32)[:k]

        def batch_search(self, queries, k):
            assert queries.shape[1] == 256
            rows = queries.shape[0]
            out = np.tile(np.array([1, 2, 3], dtype=np.uint32)[:k], (rows, 1))
            return out

    loaded = FakeLoaded()

    class FakeIndex:
        @classmethod
        def fit(cls, vectors, **kwargs):  # pylint: disable=unused-argument
            return object()

        @classmethod
        def from_prefix(cls, prefix, dram_budget_gb=1.0):  # pylint: disable=unused-argument
            return loaded

    monkeypatch.setattr(
        module,
        "laser",
        types.SimpleNamespace(Index=FakeIndex, BuildParams=_FakeBuildParams),
        raising=False,
    )
    monkeypatch.setattr(module, "_laser_runtime_supported", lambda: True, raising=False)

    adapter = module.AlayaLiteDiskLaser("l2", 256, _method_param(work_root=str(tmp_path)))
    vectors = np.random.default_rng(0).standard_normal((300, 256)).astype(np.float32)
    adapter.fit(vectors)
    adapter.set_query_arguments(88)

    query = np.random.default_rng(1).standard_normal((256,)).astype(np.float32)
    adapter.prepare_query(query, 3)
    adapter.run_prepared_query()
    assert adapter.get_prepared_query_results() == [9, 7, 3]

    batch = np.random.default_rng(2).standard_normal((4, 256)).astype(np.float32)
    adapter.batch_query(batch, 3)
    assert adapter.get_batch_results() == [[1, 2, 3]] * 4

    with pytest.raises(ValueError, match="query shape"):
        adapter.prepare_query(np.zeros((255,), dtype=np.float32), 3)
    with pytest.raises(ValueError, match="batch query shape"):
        adapter.batch_query(np.zeros((2, 255), dtype=np.float32), 3)


def test_queries_before_fit_raise(monkeypatch):
    module = _load_adapter_module(monkeypatch)
    adapter = module.AlayaLiteDiskLaser("l2", 256, _method_param())

    with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
        adapter.prepare_query(np.zeros((256,), dtype=np.float32), 3)
