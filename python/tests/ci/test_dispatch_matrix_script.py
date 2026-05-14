# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""CI meta-tests for scripts/tools/enumerate_dispatch_matrix.py + shards.cmake."""

import json
import re
import subprocess
import sys
from pathlib import Path


def test_enumerate_dispatch_matrix_matches_dispatch_header():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "tools" / "enumerate_dispatch_matrix.py"

    result = subprocess.run(
        [sys.executable, str(script), "--json"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["total_combinations"] == 390
    assert payload["dimensions"]["DataType"] == [
        "float",
        "uint8_t",
        "int8_t",
        "int32_t",
        "uint32_t",
        "double",
    ]
    assert payload["dimensions"]["IDType"] == ["uint32_t", "uint64_t"]
    assert payload["dimensions"]["DistanceType"] == ["float"]
    assert payload["dimensions"]["GraphBuilderType"] == ["HNSW", "NSG", "Fusion"]
    assert payload["dimensions"]["SearchSpaceType"] == ["Raw", "SQ8", "SQ4", "RaBitQ"]
    expected_shards = {
        "HNSW:Raw:U32": 20,
        "HNSW:Raw:U64": 20,
        "HNSW:SQ8:U32": 20,
        "HNSW:SQ8:U64": 20,
        "HNSW:SQ4:U32": 20,
        "HNSW:SQ4:U64": 20,
        "HNSW:RaBitQ:U32": 5,
        "HNSW:RaBitQ:U64": 5,
        "NSG:Raw:U32": 20,
        "NSG:Raw:U64": 20,
        "NSG:SQ8:U32": 20,
        "NSG:SQ8:U64": 20,
        "NSG:SQ4:U32": 20,
        "NSG:SQ4:U64": 20,
        "NSG:RaBitQ:U32": 5,
        "NSG:RaBitQ:U64": 5,
        "Fusion:Raw:U32": 20,
        "Fusion:Raw:U64": 20,
        "Fusion:SQ8:U32": 20,
        "Fusion:SQ8:U64": 20,
        "Fusion:SQ4:U32": 20,
        "Fusion:SQ4:U64": 20,
        "Fusion:RaBitQ:U32": 5,
        "Fusion:RaBitQ:U64": 5,
    }
    assert payload["shard_counts"] == expected_shards
    assert len(payload["shard_counts"]) == 24
    assert min(payload["shard_counts"].values()) == 5
    assert max(payload["shard_counts"].values()) == 20
    assert all("FLAT:" not in shard for shard in payload["shard_counts"])


def _parse_cmake_list(text: str, var_name: str) -> list[str]:
    match = re.search(rf"set\s*\(\s*{re.escape(var_name)}\s*(.*?)\)", text, re.DOTALL)
    assert match, f"missing CMake list: {var_name}"
    body = match.group(1)
    return [token.strip() for token in re.split(r"\s+", body) if token.strip()]


def test_instantiation_shards_are_generated_from_manifest():
    repo_root = Path(__file__).resolve().parents[3]
    inst_dir = repo_root / "python" / "src" / "instantiations"
    generated_dir = repo_root / "build" / "python" / "generated" / "instantiations"
    cmake_lists = (repo_root / "python" / "CMakeLists.txt").read_text(encoding="utf-8")
    shards_cmake = (repo_root / "python" / "cmake" / "shards.cmake").read_text(encoding="utf-8")

    assert not inst_dir.exists()
    assert "src/instantiations/" not in cmake_lists
    assert "include(cmake/shards.cmake)" in cmake_lists
    assert "include(cmake/generate_shards.cmake)" in cmake_lists

    shard_names = _parse_cmake_list(shards_cmake, "ALAYA_PYINDEX_SHARDS")
    assert len(shard_names) == 24

    strategy_match = re.search(r'set\(\s*ALAYA_PYINDEX_SHARD_STRATEGY\s+"([^"]+)"\s*\)', shards_cmake)
    assert strategy_match, "ALAYA_PYINDEX_SHARD_STRATEGY scalar not found in shards.cmake"
    uniform_strategy = strategy_match.group(1)
    assert uniform_strategy in {"core_and_search", "single_tu"}

    expected_generated: set[str] = set()
    for shard_name in shard_names:
        if uniform_strategy == "core_and_search":
            expected_generated.add(f"inst_{shard_name}_core.cpp")
            expected_generated.add(f"inst_{shard_name}_search.cpp")
        else:
            expected_generated.add(f"inst_{shard_name}.cpp")

    if generated_dir.exists():
        actual_generated = {path.name for path in generated_dir.glob("inst_*.cpp")}
        assert actual_generated == expected_generated
