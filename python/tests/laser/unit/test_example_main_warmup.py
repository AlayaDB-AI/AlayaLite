# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for LASER example warmup dispatch."""

# pylint: disable=protected-access

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_example_main():
    root = Path(__file__).resolve().parents[4]
    path = root / "examples" / "laser" / "main.py"
    spec = importlib.util.spec_from_file_location("example_laser_main", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeIndex:
    """Minimal index stub that records warmup calls."""

    def __init__(self) -> None:
        self.search_calls = 0
        self.batch_search_calls = 0

    def search(self, query: np.ndarray, topk: int) -> list[int]:
        self.search_calls += 1
        return [int(topk), int(query.shape[0])]

    def batch_search(self, queries: np.ndarray, topk: int) -> np.ndarray:
        self.batch_search_calls += 1
        return np.zeros((queries.shape[0], topk), dtype=np.uint32)


def test_single_thread_warmup_uses_search_path() -> None:
    main = _load_example_main()
    index = FakeIndex()
    queries = np.zeros((3, 4), dtype=np.float32)

    main._warmup_index(index, queries, topk=2, rounds=2, single_search=True)

    assert index.search_calls == 6
    assert index.batch_search_calls == 0


def test_batch_warmup_uses_batch_search_path() -> None:
    main = _load_example_main()
    index = FakeIndex()
    queries = np.zeros((3, 4), dtype=np.float32)

    main._warmup_index(index, queries, topk=2, rounds=2, single_search=False)

    assert index.search_calls == 0
    assert index.batch_search_calls == 2
