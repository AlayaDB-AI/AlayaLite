#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Enumerate the legal PyIndex dispatch matrix from python/include/instantiations/dispatch.hpp.

Parses the DISPATCH_* macros in dispatch.hpp, computes the Cartesian product
of (DataType, IDType, DistanceType, SearchScalarDataType, BuildScalarDataType,
Graph, Quant) with the if-constexpr prunings (RaBitQ is float-only, FLAT is
not admitted), and reports every legal combination plus the 24-shard grouping.

When to run:
- Before/after editing dispatch.hpp to quantify the change in combination count
- When writing a PR description ("this change goes from 390 to X combinations")
- When debugging a missing template instantiation
- Consumed by python/tests/ci/test_dispatch_matrix_script.py in CI

Usage:
    uv run --no-project python scripts/tools/enumerate_dispatch_matrix.py
    uv run --no-project python scripts/tools/enumerate_dispatch_matrix.py --json
    uv run --no-project python scripts/tools/enumerate_dispatch_matrix.py \\
        --dispatch-hpp PATH

Output:
    Default text mode prints: dispatch_hpp, total_combinations, dimensions,
    shard_counts, combinations.
    --json: same fields as machine-readable JSON, suitable for `jq` queries.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISPATCH_HPP = REPO_ROOT / "python" / "include" / "instantiations" / "dispatch.hpp"


def _macro_body(text: str, name: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"#define {name}"):
            body = [line]
            cursor = index
            while body[-1].rstrip().endswith("\\"):
                cursor += 1
                body.append(lines[cursor])
            return "\n".join(item.rstrip().removesuffix("\\").rstrip() for item in body)
    raise ValueError(f"macro not found: {name}")


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _parse_dtype_list(text: str) -> list[str]:
    body = _macro_body(text, "DISPATCH_DATA_TYPE")
    return _unique(re.findall(r"py::dtype::of<([^>]+)>", body))


def _parse_id_type_list(text: str) -> list[str]:
    body = _macro_body(text, "DISPATCH_ID_TYPE")
    return _unique(re.findall(r"py::dtype::of<([^>]+)>", body))


def _parse_using_type_list(text: str, macro_name: str, alias_name: str) -> list[str]:
    body = _macro_body(text, macro_name)
    return _unique(re.findall(rf"using {alias_name} = ([^;]+);", body))


def _parse_graph_builder_names(text: str) -> list[str]:
    body = _macro_body(text, "DISPATCH_BUILDER_TYPE")
    names: list[str] = []
    if "HNSWBuilder<BuildSpaceType>" in body:
        names.append("HNSW")
    if "NSGBuilder<BuildSpaceType>" in body:
        names.append("NSG")
    if "FusionGraphBuilder<BuildSpaceType" in body:
        names.append("Fusion")
    return names


def _parse_search_space_names(text: str) -> list[str]:
    body = _macro_body(text, "DISPATCH_SEARCH_SPACE_TYPE")
    result: list[str] = []
    for quantization in re.findall(r"QuantizationType::([A-Z0-9]+)", body):
        if quantization == "NONE":
            result.append("Raw")
        elif quantization == "SQ8":
            result.append("SQ8")
        elif quantization == "SQ4":
            result.append("SQ4")
        elif quantization == "RABITQ":
            result.append("RaBitQ")
        else:
            raise ValueError(f"unknown quantization branch: {quantization}")
    return _unique(result)


def _build_space_raw_search_scalar(data_type: str, id_type: str, search_scalar_type: str) -> str:
    return f"RawSpace<{data_type}, float, {id_type}, SequentialStorage<{data_type}, {id_type}>, {search_scalar_type}>"


def _build_space_raw_empty_scalar(data_type: str, id_type: str) -> str:
    return f"RawSpace<{data_type}, float, {id_type}, SequentialStorage<{data_type}, {id_type}>, EmptyScalarData>"


def _build_space_rabitq(id_type: str, search_scalar_type: str) -> str:
    return f"RaBitQSpace<float, float, {id_type}, {search_scalar_type}>"


def _search_space(data_type: str, id_type: str, search_scalar_type: str, quantization: str) -> str:
    if quantization == "NONE":
        return (
            f"RawSpace<{data_type}, float, {id_type}, SequentialStorage<{data_type}, {id_type}>, {search_scalar_type}>"
        )
    if quantization == "SQ8":
        return f"SQ8Space<{data_type}, float, {id_type}, SequentialStorage<uint8_t, {id_type}>, {search_scalar_type}>"
    if quantization == "SQ4":
        return f"SQ4Space<{data_type}, float, {id_type}, SequentialStorage<uint8_t, {id_type}>, {search_scalar_type}>"
    if quantization == "RABITQ":
        return f"RaBitQSpace<float, float, {id_type}, {search_scalar_type}>"
    raise ValueError(f"unknown quantization: {quantization}")


def _graph_builder(graph_name: str, build_space_type: str) -> str:
    if graph_name == "HNSW":
        return f"HNSWBuilder<{build_space_type}>"
    if graph_name == "NSG":
        return f"NSGBuilder<{build_space_type}>"
    if graph_name == "Fusion":
        return (
            f"FusionGraphBuilder<{build_space_type}, HNSWBuilder<{build_space_type}>, NSGBuilder<{build_space_type}>>"
        )
    raise ValueError(f"unknown graph builder: {graph_name}")


def _shard_suffix(id_type: str) -> str:
    if id_type == "uint32_t":
        return "U32"
    if id_type == "uint64_t":
        return "U64"
    raise ValueError(f"unknown ID type: {id_type}")


def enumerate_dispatch_matrix(dispatch_hpp: Path) -> dict[str, object]:
    text = dispatch_hpp.read_text(encoding="utf-8")
    data_types = _parse_dtype_list(text)
    id_types = _parse_id_type_list(text)
    distance_types = _parse_using_type_list(text, "DISPATCH_DISTANCE_TYPE", "DistanceType")
    search_scalar_types = _parse_using_type_list(text, "DISPATCH_SEARCH_SCALAR_TYPE", "SearchScalarDataType")
    build_scalar_types = _parse_using_type_list(text, "DISPATCH_BUILD_SCALAR_TYPE", "BuildScalarDataType")
    graph_names = _parse_graph_builder_names(text)
    search_space_names = _parse_search_space_names(text)

    combinations: list[dict[str, str]] = []
    for data_type in data_types:
        for id_type in id_types:
            for search_scalar_type in search_scalar_types:
                if search_scalar_type == "ScalarData":
                    build_variants = [
                        (
                            "RAW_SEARCH_SCALAR",
                            _build_space_raw_search_scalar(data_type, id_type, search_scalar_type),
                        ),
                        ("RAW_EMPTY_SCALAR", _build_space_raw_empty_scalar(data_type, id_type)),
                    ]
                    if data_type == "float":
                        build_variants.append(("RABITQ", _build_space_rabitq(id_type, search_scalar_type)))
                else:
                    build_variants = [("RAW_EMPTY_SCALAR", _build_space_raw_empty_scalar(data_type, id_type))]
                    if data_type == "float":
                        build_variants.append(("RABITQ", _build_space_rabitq(id_type, search_scalar_type)))

                search_variants = [
                    ("Raw", "NONE", _search_space(data_type, id_type, search_scalar_type, "NONE")),
                    ("SQ8", "SQ8", _search_space(data_type, id_type, search_scalar_type, "SQ8")),
                    ("SQ4", "SQ4", _search_space(data_type, id_type, search_scalar_type, "SQ4")),
                ]
                if data_type == "float":
                    search_variants.append(
                        (
                            "RaBitQ",
                            "RABITQ",
                            _search_space(data_type, id_type, search_scalar_type, "RABITQ"),
                        )
                    )

                for build_variant, build_space_type in build_variants:
                    for search_space_name, quantization, search_space_type in search_variants:
                        for graph_name in graph_names:
                            combinations.append(
                                {
                                    "DataType": data_type,
                                    "IDType": id_type,
                                    "DistanceType": distance_types[0],
                                    "SearchScalarDataType": search_scalar_type,
                                    "BuildScalarDataType": build_scalar_types[0],
                                    "BuildVariant": build_variant,
                                    "BuildSpaceType": build_space_type,
                                    "GraphBuilderType": _graph_builder(graph_name, build_space_type),
                                    "SearchSpaceType": search_space_type,
                                    "IndexType": graph_name,
                                    "QuantizationType": quantization,
                                    "shard": f"{graph_name}:{search_space_name}:{_shard_suffix(id_type)}",
                                }
                            )

    shard_counts = Counter(item["shard"] for item in combinations)
    return {
        "dispatch_hpp": str(dispatch_hpp),
        "dimensions": {
            "DataType": data_types,
            "IDType": id_types,
            "DistanceType": distance_types,
            "SearchScalarDataType": search_scalar_types,
            "BuildScalarDataType": build_scalar_types,
            "GraphBuilderType": graph_names,
            "SearchSpaceType": search_space_names,
        },
        "total_combinations": len(combinations),
        "shard_counts": dict(sorted(shard_counts.items())),
        "combinations": combinations,
    }


def _format_text(payload: dict[str, object]) -> str:
    lines = [
        f"dispatch_hpp: {payload['dispatch_hpp']}",
        f"total_combinations: {payload['total_combinations']}",
        "dimensions:",
    ]
    dimensions = payload["dimensions"]
    assert isinstance(dimensions, dict)
    for name, values in dimensions.items():
        assert isinstance(values, list)
        lines.append(f"  {name}: {', '.join(values)}")
    lines.append("shard_counts:")
    shard_counts = payload["shard_counts"]
    assert isinstance(shard_counts, dict)
    for shard, count in shard_counts.items():
        lines.append(f"  {shard}: {count}")
    lines.append("combinations:")
    combinations = payload["combinations"]
    assert isinstance(combinations, list)
    for combo in combinations:
        assert isinstance(combo, dict)
        lines.append(
            "  "
            f"{combo['IndexType']} {combo['QuantizationType']} "
            f"{combo['DataType']} {combo['IDType']} "
            f"{combo['SearchScalarDataType']} -> "
            f"{combo['GraphBuilderType']} | {combo['SearchSpaceType']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dispatch-hpp",
        type=Path,
        default=DEFAULT_DISPATCH_HPP,
        help="Path to python/include/instantiations/dispatch.hpp",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    payload = enumerate_dispatch_matrix(args.dispatch_hpp)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_text(payload))


if __name__ == "__main__":
    main()
