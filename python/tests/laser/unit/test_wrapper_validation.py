# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the high-level LASER Python wrapper."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


class _FakeRawSearch:
    """Minimal stand-in for the pybind RawIndex search interface."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...], int]] = []

    def search(self, query: np.ndarray, k: int):
        self.calls.append(("search", tuple(query.shape), int(k)))
        return np.arange(k, dtype=np.uint32)

    def batch_search(self, queries: np.ndarray, k: int):
        self.calls.append(("batch_search", tuple(queries.shape), int(k)))
        return np.zeros((queries.shape[0], k), dtype=np.uint32)


def _loaded_index(raw: _FakeRawSearch, *, raw_dim: int = 4):
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    return laser.Index(
        raw=raw,
        prefix="unit",
        params=laser._IndexParams(  # pylint: disable=protected-access
            metric="l2",
            n=10,
            raw_dim=raw_dim,
            main_dim=raw_dim,
            R=2,
            prefix="unit",
        ),
        loaded=True,
    )


def test_search_rejects_wrong_width_before_native_call() -> None:
    raw = _FakeRawSearch()
    idx = _loaded_index(raw, raw_dim=4)

    with pytest.raises(ValueError, match=r"query\.shape\[0\]=3"):
        idx.search(np.zeros(3, dtype=np.float32), k=1)

    assert not raw.calls


def test_batch_search_rejects_wrong_width_before_native_call() -> None:
    raw = _FakeRawSearch()
    idx = _loaded_index(raw, raw_dim=4)

    with pytest.raises(ValueError, match=r"queries\.shape\[1\]=3"):
        idx.batch_search(np.zeros((2, 3), dtype=np.float32), k=1)

    assert not raw.calls


def test_fit_resolves_zero_threads_before_raw_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import alayalite  # pylint: disable=import-outside-toplevel
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    constructed = []

    class FakeRawIndex:  # pylint: disable=too-few-public-methods
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.build_call = None
            constructed.append(self)

        def build_index(self, **kwargs) -> None:
            self.build_call = kwargs

    fake_raw_mod = types.SimpleNamespace(Index=FakeRawIndex)
    fake_vamana_mod = types.SimpleNamespace(build_index=lambda **_kwargs: None)

    monkeypatch.setattr(laser, "_raw_laser_mod", fake_raw_mod)  # pylint: disable=protected-access
    monkeypatch.setattr(laser.os, "cpu_count", lambda: 7)
    monkeypatch.setitem(sys.modules, "alayalite.vamana", fake_vamana_mod)
    monkeypatch.setattr(alayalite, "vamana", fake_vamana_mod, raising=False)

    vectors = np.zeros((4, 128), dtype=np.float32)
    laser.Index.fit(
        vectors,
        output_dir=tmp_path,
        name="threads",
        build_params=laser.BuildParams(main_dim=128, R=2, L=2, disable_medoid=True),
        num_threads=0,
        auto_load=False,
    )

    assert len(constructed) == 1
    assert constructed[0].build_call["num_thread"] == 7


def test_fit_removes_mismatched_seed_sidecar_before_rebuild_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import alayalite  # pylint: disable=import-outside-toplevel
    from alayalite import laser  # pylint: disable=import-outside-toplevel
    from alayalite.laser._idempotence import seed_sidecar_path  # pylint: disable=import-outside-toplevel

    def fail_build_index(**_kwargs) -> None:
        raise RuntimeError("vamana boom")

    fake_vamana_mod = types.SimpleNamespace(build_index=fail_build_index)
    monkeypatch.setitem(sys.modules, "alayalite.vamana", fake_vamana_mod)
    monkeypatch.setattr(alayalite, "vamana", fake_vamana_mod, raising=False)

    prefix = tmp_path / "seed"
    sidecar = Path(seed_sidecar_path(str(prefix)))
    sidecar.write_text("42\n", encoding="utf-8")

    vectors = np.zeros((4, 128), dtype=np.float32)
    with pytest.raises(RuntimeError, match="vamana boom"):
        laser.Index.fit(
            vectors,
            output_dir=tmp_path,
            name="seed",
            build_params=laser.BuildParams(main_dim=128, R=2, L=2, disable_medoid=True),
            seed=43,
            num_threads=1,
            skip_existing=True,
            auto_load=False,
        )

    assert not sidecar.exists()
