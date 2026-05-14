#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Verify the physical layout of pybind binding files under python/src/.

Checks:
- python/src/ root contains only pybind.cpp
- python/src/bindings/ has the 7 required pybind_index*.cpp files
- python/src/submodules/ has pybind_{disk,vamana,laser}.cpp, no cross-mixing
- pybind.cpp wires every submodule and top-level register_*
- pybind_index.cpp registers Client / IndexParams / PyIndexInterface on m
- bindings/README.md ownership table is in sync with the code

When to run:
- After adding, moving, or renaming a binding cpp file
- After adding a new submodule
- In CI on every PR (pybind-static-checks job)

Usage:
    uv run --no-project python scripts/ci/pybind_layout.py

Exit code:
    0  layout consistent
    1  one or more violations printed as "ERROR: <message>"
"""

from __future__ import annotations

from pathlib import Path


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    python_root = repo_root / "python"
    src_root = python_root / "src"
    bindings_dir = src_root / "bindings"
    submodules_dir = src_root / "submodules"
    include_bindings_header = python_root / "include" / "bindings" / "pybind_index_methods.hpp"
    legacy_include_header = python_root / "include" / "pybind_index_methods.hpp"
    readme_path = bindings_dir / "README.md"

    errors: list[str] = []

    root_cpp_files = sorted(p.name for p in src_root.glob("*.cpp"))
    _require(
        root_cpp_files == ["pybind.cpp"],
        f"python/src root must only contain pybind.cpp, got: {root_cpp_files}",
        errors,
    )

    required_bindings = {
        "pybind_index.cpp",
        "pybind_index_search.cpp",
        "pybind_index_fit.cpp",
        "pybind_index_mutate.cpp",
        "pybind_index_scalar.cpp",
        "pybind_index_hybrid.cpp",
        "pybind_index_io.cpp",
    }
    required_submodules = {"pybind_disk.cpp", "pybind_vamana.cpp", "pybind_laser.cpp"}

    binding_cpp_files = {p.name for p in bindings_dir.glob("*.cpp")}
    submodule_cpp_files = {p.name for p in submodules_dir.glob("*.cpp")}

    _require(
        required_bindings.issubset(binding_cpp_files),
        f"missing files in python/src/bindings: {sorted(required_bindings - binding_cpp_files)}",
        errors,
    )
    _require(
        required_submodules.issubset(submodule_cpp_files),
        f"missing files in python/src/submodules: {sorted(required_submodules - submodule_cpp_files)}",
        errors,
    )
    _require(
        not (required_submodules & binding_cpp_files),
        f"submodule shells must not live in bindings/: {sorted(required_submodules & binding_cpp_files)}",
        errors,
    )
    _require(
        include_bindings_header.exists(),
        "python/include/bindings/pybind_index_methods.hpp must exist",
        errors,
    )
    _require(
        not legacy_include_header.exists(),
        "legacy python/include/pybind_index_methods.hpp must not exist",
        errors,
    )

    pybind_cpp_text = (src_root / "pybind.cpp").read_text(encoding="utf-8")
    pybind_index_cpp_text = (bindings_dir / "pybind_index.cpp").read_text(encoding="utf-8")
    disk_header_text = (python_root / "include" / "disk_collection.hpp").read_text(encoding="utf-8")

    _require('def_submodule("vamana"' in pybind_cpp_text, "pybind.cpp must define vamana submodule", errors)
    _require("register_vamana" in pybind_cpp_text, "pybind.cpp must register vamana submodule", errors)
    _require('def_submodule("laser"' in pybind_cpp_text, "pybind.cpp must define laser submodule", errors)
    _require("register_laser" in pybind_cpp_text, "pybind.cpp must register laser submodule", errors)
    _require("register_index(m)" in pybind_cpp_text, "pybind.cpp must register top-level index bindings", errors)
    _require("register_disk(m)" in pybind_cpp_text, "pybind.cpp must register top-level disk bindings", errors)

    _require(
        'py::class_<Client>(m, "Client")' in pybind_index_cpp_text,
        "Client must be registered on top-level module m",
        errors,
    )
    _require(
        'py::class_<IndexParams>(m, "IndexParams")' in pybind_index_cpp_text,
        "IndexParams must be registered on top-level module m",
        errors,
    )
    _require(
        'PyIndexInterfaceClass cls(m, "PyIndexInterface")' in pybind_index_cpp_text,
        "PyIndexInterface must be registered on top-level module m",
        errors,
    )
    _require(
        '(m, "DiskCollection")' in disk_header_text,
        "DiskCollection must be registered on top-level module m",
        errors,
    )

    try:
        readme = readme_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append("python/src/bindings/README.md must exist")
    else:
        required_entries = [
            "alayalite.Client",
            "alayalite.IndexParams",
            "alayalite.PyIndexInterface",
            "alayalite.DiskCollection",
            "alayalite.vamana.build_index",
            "alayalite.laser.Index",
        ]
        for entry in required_entries:
            _require(entry in readme, f"README missing ownership entry: {entry}", errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("pybind layout check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
