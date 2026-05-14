# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Smoke test the 24 top-level pybind dispatch shards."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from alayalite import Client

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

_GRAPH_TYPES: Final[tuple[str, ...]] = ("hnsw", "nsg", "fusion")
_QUANT_TYPES: Final[tuple[str, ...]] = ("none", "sq8", "sq4", "rabitq")
_ID_TYPES: Final[tuple[type[np.unsignedinteger], ...]] = (np.uint32, np.uint64)

_GRAPH_SYMBOL_TOKENS: Final[dict[str, str]] = {
    "hnsw": "HNSWBuilder<",
    "nsg": "NSGBuilder<",
    "fusion": "FusionGraphBuilder<",
}
_QUANT_SYMBOL_TOKENS: Final[dict[str, str]] = {
    "none": "RawSpace<",
    "sq8": "SQ8Space<",
    "sq4": "SQ4Space<",
    "rabitq": "RaBitQSpace<",
}
_ID_SYMBOL_CANDIDATES: Final[dict[type[np.unsignedinteger], tuple[str, ...]]] = {
    np.uint32: ("uint32_t", "unsigned int"),
    np.uint64: ("uint64_t", "unsigned long", "unsigned long long"),
}


def _find_extension_path() -> Path:
    candidates = sorted(REPO_ROOT.glob("build/**/_alayalitepy*.so"))
    if not candidates:
        raise FileNotFoundError("Cannot find built extension under build/**/_alayalitepy*.so")
    return candidates[0]


def _load_defined_symbols() -> list[str]:
    ext_path = _find_extension_path()
    result = subprocess.run(
        ["nm", "-C", "--defined-only", str(ext_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


@pytest.fixture(scope="module")
def defined_symbols() -> list[str]:
    return _load_defined_symbols()


def _dispatch_cases() -> list[tuple[str, str, type[np.unsignedinteger]]]:
    return [
        (graph, quantization, id_type)
        for graph in _GRAPH_TYPES
        for quantization in _QUANT_TYPES
        for id_type in _ID_TYPES
    ]


_DISPATCH_CASES: Final[list[tuple[str, str, type[np.unsignedinteger]]]] = _dispatch_cases()
_DISPATCH_CASE_IDS: Final[list[str]] = [
    f"{graph}-{quantization}-{np.dtype(id_type).name}" for graph, quantization, id_type in _DISPATCH_CASES
]


@pytest.mark.parametrize(("graph", "quantization", "id_type"), _DISPATCH_CASES, ids=_DISPATCH_CASE_IDS)
def test_dispatch_matrix_smoke(  # pylint: disable=redefined-outer-name
    graph: str,
    quantization: str,
    id_type: type[np.unsignedinteger],
    defined_symbols: list[str],
) -> None:
    rng = np.random.default_rng(20260513)
    dim = 128 if quantization == "rabitq" else 16
    vectors = rng.random((100, dim), dtype=np.float32)
    query = rng.random(dim, dtype=np.float32)

    client = Client()
    index_name = f"dispatch_{graph}_{quantization}_{np.dtype(id_type).name}"
    index = client.create_index(
        name=index_name,
        index_type=graph,
        quantization_type=quantization,
        id_type=id_type,
        metric="l2",
    )
    index.fit(vectors, ef_construction=32, num_threads=1)
    result = index.search(query, topk=5, ef_search=16).reshape(1, -1)

    assert result.shape == (1, 5)

    graph_token = _GRAPH_SYMBOL_TOKENS[graph]
    quant_token = _QUANT_SYMBOL_TOKENS[quantization]
    id_candidates = _ID_SYMBOL_CANDIDATES[id_type]
    has_symbol = any(
        ("PyIndex<" in line)
        and (graph_token in line)
        and (quant_token in line)
        and any(id_token in line for id_token in id_candidates)
        for line in defined_symbols
    )
    assert has_symbol, f"missing PyIndex symbol for {graph}/{quantization}/{np.dtype(id_type).name}"
