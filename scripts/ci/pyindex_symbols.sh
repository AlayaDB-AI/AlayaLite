#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Verify paired inst_<shard>_core.cpp.o / inst_<shard>_search.cpp.o object
# files have *disjoint* alaya::PyIndex<...> symbol sets. Overlap means the
# same template method body was instantiated in both TUs, which causes
# duplicate-symbol link errors or non-deterministic linker resolution.
#
# When to run:
# - After editing python/include/instantiations/pyindex_instantiation_core.hpp or _search.hpp
# - After moving a PyIndex method between the core and search halves
# - In CI (py-unit-test job, after build) and locally after a clean build
#
# Usage:
#     bash scripts/ci/pyindex_symbols.sh [build_dir]
#     # default build_dir = "build"; pass "." to search the whole worktree
#
# Limitation:
#     Only inspects shards built with split_strategy=core_and_search. With
#     split_strategy=single_tu no inst_*_core.cpp.o exists and the script
#     exits 2.
#
# Exit code:
#     0  no overlap (all shard pairs clean)
#     1  overlap detected (offending symbols printed)
#     2  build_dir missing or no inst_*_core.cpp.o found

set -euo pipefail

build_dir="${1:-build}"
if [[ ! -d "${build_dir}" ]]; then
  echo "error: build directory not found: ${build_dir}" >&2
  exit 2
fi

mapfile -t core_objects < <(find "${build_dir}" -type f -name "inst_*_core.cpp.o" | sort)
if [[ ${#core_objects[@]} -eq 0 ]]; then
  echo "error: no core shard object files found under ${build_dir}" >&2
  exit 2
fi

status=0
checked=0

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

extract_pyindex_symbols() {
  local obj="$1"
  local out="$2"

  nm -C --defined-only "${obj}" \
    | awk '
        /^[0-9a-fA-F]+ [[:alpha:]] / {
          $1 = ""
          $2 = ""
          sub(/^ +/, "", $0)
          if ($0 ~ /^alaya::PyIndex</) {
            print $0
          }
        }
      ' \
    | sort -u > "${out}"
}

for core_obj in "${core_objects[@]}"; do
  core_name="$(basename "${core_obj}")"
  shard="${core_name#inst_}"
  shard="${shard%_core.cpp.o}"
  search_obj="$(dirname "${core_obj}")/inst_${shard}_search.cpp.o"

  if [[ ! -f "${search_obj}" ]]; then
    echo "error: missing search object for shard ${shard}: ${search_obj}" >&2
    status=1
    continue
  fi

  core_syms="${tmpdir}/${shard}.core.txt"
  search_syms="${tmpdir}/${shard}.search.txt"
  overlap_syms="${tmpdir}/${shard}.overlap.txt"

  extract_pyindex_symbols "${core_obj}" "${core_syms}"
  extract_pyindex_symbols "${search_obj}" "${search_syms}"

  comm -12 "${core_syms}" "${search_syms}" > "${overlap_syms}" || true

  if [[ -s "${overlap_syms}" ]]; then
    echo "overlap detected for shard ${shard}" >&2
    cat "${overlap_syms}" >&2
    status=1
  fi

  checked=$((checked + 1))
done

if [[ ${status} -ne 0 ]]; then
  echo "checked ${checked} core/search shard pairs; overlap check failed" >&2
  exit ${status}
fi

echo "checked ${checked} core/search shard pairs; no overlaps detected"
