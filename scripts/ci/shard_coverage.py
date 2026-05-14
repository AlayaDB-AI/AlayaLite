#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Verify instantiations.hpp macro lattice and shards.cmake manifest agree.

Uses the C++ preprocessor (gcc -E -P -nostdinc -x c++ -) to expand
ALAYA_PYINDEX_INSTANTIATIONS(X), captures every (GraphBuilder, SearchSpace)
pair the build will actually see, derives the 24 shard names, and compares
against ALAYA_PYINDEX_SHARDS in shards.cmake. Any mismatch is a build-time
drift that would lead to silent under/over instantiation.

When to run:
- After editing python/include/instantiations/instantiations.hpp
- After editing python/cmake/shards.cmake
- After editing python/include/instantiations/dispatch.hpp (which feeds instantiations)
- In CI on every PR (pybind-static-checks job)

Usage:
    uv run --no-project python scripts/ci/shard_coverage.py

Requires:
    gcc with C++ frontend on PATH (preprocessor only; no link)

Exit code:
    0  shard sets match (A == B == 24 shards)
    1  shard set mismatch; prints A - B and B - A diffs
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANTIATIONS = REPO_ROOT / "python" / "include" / "instantiations" / "instantiations.hpp"
DEFAULT_SHARDS_CMAKE = REPO_ROOT / "python" / "cmake" / "shards.cmake"

_CAPTURE_NAME = "ALAYA_CAPTURE_PAIR"


def _strip_includes_and_externs(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#include "):
            continue
        if "ALAYA_DECLARE_EXTERN" in stripped:
            continue
        lines.append(line)
    return "\n".join(lines)


def _expand_instantiations(instantiations_path: Path) -> str:
    source = instantiations_path.read_text(encoding="utf-8")
    source = _strip_includes_and_externs(source)
    snippet = "\n".join(
        [
            source,
            f"#define ALAYA_CAPTURE(GraphBuilderT, SearchSpaceT) {_CAPTURE_NAME}(GraphBuilderT, SearchSpaceT)",
            "ALAYA_PYINDEX_INSTANTIATIONS(ALAYA_CAPTURE)",
            "",
        ]
    )

    result = subprocess.run(
        ["gcc", "-E", "-P", "-nostdinc", "-x", "c++", "-"],
        input=snippet,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _split_top_level(text: str) -> tuple[str, str]:
    depth = 0
    split_at = -1
    for index, char in enumerate(text):
        if char in "(<[{":
            depth += 1
        elif char in ")>]}":
            depth -= 1
        elif char == "," and depth == 0:
            split_at = index
            break
    if split_at < 0:
        raise ValueError("cannot split top-level args")
    return text[:split_at].strip(), text[split_at + 1 :].strip()


def _parse_capture_pairs(expanded: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    marker = f"{_CAPTURE_NAME}("
    cursor = 0
    while True:
        start = expanded.find(marker, cursor)
        if start < 0:
            break
        body_start = start + len(marker)
        depth = 1
        pos = body_start
        while pos < len(expanded) and depth > 0:
            char = expanded[pos]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            pos += 1
        if depth != 0:
            raise ValueError("unbalanced ALAYA_CAPTURE_PAIR invocation")

        body = expanded[body_start : pos - 1].strip()
        graph_type, search_type = _split_top_level(body)
        pairs.append((graph_type, search_type))
        cursor = pos
    return pairs


def _graph_name(graph_type: str) -> str:
    if "FusionGraphBuilder<" in graph_type:
        return "fusion"
    if "HNSWBuilder<" in graph_type:
        return "hnsw"
    if "NSGBuilder<" in graph_type:
        return "nsg"
    raise ValueError(f"unknown graph builder type: {graph_type}")


def _quant_name(search_type: str) -> str:
    if "RaBitQSpace<" in search_type:
        return "rabitq"
    if "SQ8Space<" in search_type:
        return "sq8"
    if "SQ4Space<" in search_type:
        return "sq4"
    if "RawSpace<" in search_type:
        return "raw"
    raise ValueError(f"unknown search space type: {search_type}")


def _id_suffix(text: str) -> str:
    if "uint64_t" in text or "unsigned long long" in text or "long unsigned int" in text:
        return "u64"
    if "uint32_t" in text or "unsigned int" in text:
        return "u32"
    raise ValueError(f"cannot infer id type from: {text}")


def _pairs_to_shards(pairs: list[tuple[str, str]]) -> set[str]:
    shards: set[str] = set()
    for graph_type, search_type in pairs:
        shards.add(f"{_graph_name(graph_type)}_{_quant_name(search_type)}_{_id_suffix(search_type)}")
    return shards


def _parse_cmake_list(text: str, name: str) -> list[str]:
    pattern = re.compile(rf"set\s*\(\s*{re.escape(name)}\s*(.*?)\)", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return []
    body = match.group(1)
    items = []
    for token in re.split(r"\s+", body):
        token = token.strip().strip('"').strip("'")
        if not token or token.startswith("#"):
            continue
        if token == ")":
            continue
        items.append(token)
    return items


def _macro_to_shard(macro_name: str) -> str:
    match = re.fullmatch(r"ALAYA_PYINDEX_([A-Z0-9_]+)_INSTANTIATIONS", macro_name)
    if not match:
        raise ValueError(f"invalid macro name: {macro_name}")
    return match.group(1).lower()


def _load_manifest_shards(shards_cmake: Path) -> set[str]:
    text = shards_cmake.read_text(encoding="utf-8")
    shard_names = _parse_cmake_list(text, "ALAYA_PYINDEX_SHARDS")
    if shard_names:
        return {name.lower() for name in shard_names}

    macro_names = _parse_cmake_list(text, "ALAYA_PYINDEX_SHARD_MACROS")
    if macro_names:
        return {_macro_to_shard(macro_name) for macro_name in macro_names}

    raise ValueError("shards.cmake found but neither ALAYA_PYINDEX_SHARDS nor ALAYA_PYINDEX_SHARD_MACROS exists")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instantiations", type=Path, default=DEFAULT_INSTANTIATIONS)
    parser.add_argument("--shards-cmake", type=Path, default=DEFAULT_SHARDS_CMAKE)
    args = parser.parse_args()

    expanded = _expand_instantiations(args.instantiations)
    pairs = _parse_capture_pairs(expanded)
    shards_a = _pairs_to_shards(pairs)
    print(f"[instantiations] expanded pairs: {len(pairs)}")
    print(f"[instantiations] shard set size: {len(shards_a)}")

    if len(shards_a) != 24:
        print(f"ERROR: expected 24 shards from instantiations, got {len(shards_a)}", file=sys.stderr)
        return 1

    if not args.shards_cmake.exists():
        print("[manifest] shards.cmake not found; validated instantiations shard set only")
        return 0

    shards_b = _load_manifest_shards(args.shards_cmake)
    print(f"[manifest] shard set size: {len(shards_b)}")
    if shards_a != shards_b:
        only_a = sorted(shards_a - shards_b)
        only_b = sorted(shards_b - shards_a)
        print("ERROR: shard set mismatch", file=sys.stderr)
        print(f"  A - B ({len(only_a)}): {only_a}", file=sys.stderr)
        print(f"  B - A ({len(only_b)}): {only_b}", file=sys.stderr)
        return 1

    print("OK: shard manifest coverage check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
