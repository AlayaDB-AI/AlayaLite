#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Measure per-shard compile time of _alayalitepy by parsing ninja's
# .ninja_log after a clean rebuild. Output CSV drives sharding strategy
# decisions (core_and_search vs single_tu). The CSVs already in
# python/cmake/ are reference data only — CMake does NOT consume them.
#
# When to run:
# - Before/after switching shard split_strategy (compare wall-clocks)
# - After adding new shards (find unexpectedly slow ones)
# - After header refactors that should reduce TU compile time
#
# Usage:
#     bash scripts/bench/shard_compile_time.sh [build_dir] [output_csv]
#     # default build_dir = build-ninja
#     # default output_csv = python/cmake/baseline_compile_times.csv
#     PARALLEL_JOBS=64 TARGET_NAME=_alayalitepy bash <above>   # tune ninja
#
# Cluster note (Slurm):
#     This script runs --clean-first and rebuilds the wheel; on the login
#     node it may be killed. Use srun on a compute node:
#         srun --pty -p compute -c 64 \
#             bash scripts/bench/shard_compile_time.sh
#
# Output:
#     CSV columns: shard_name, strategy, wall_clock_ms
#     Stdout: total_wall_clock_ms = max(end) - min(start), i.e. ninja's
#     parallel wall-clock (not the sum of per-shard times).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUILD_DIR="${1:-${REPO_ROOT}/build-ninja}"
OUTPUT_CSV="${2:-${REPO_ROOT}/python/cmake/baseline_compile_times.csv}"
TARGET_NAME="${TARGET_NAME:-_alayalitepy}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc)}"

mkdir -p "$(dirname "${OUTPUT_CSV}")"

if [[ ! -f "${BUILD_DIR}/build.ninja" ]]; then
  cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G Ninja
fi

rm -f "${BUILD_DIR}/.ninja_log"
cmake --build "${BUILD_DIR}" --target "${TARGET_NAME}" --clean-first --parallel "${PARALLEL_JOBS}"

if [[ ! -f "${BUILD_DIR}/.ninja_log" ]]; then
  echo "error: missing ${BUILD_DIR}/.ninja_log after build" >&2
  exit 1
fi

python3 - "$BUILD_DIR/.ninja_log" "$OUTPUT_CSV" <<'PY'
import csv
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])

pattern = re.compile(
    r"python/CMakeFiles/_alayalitepy\.dir/(?:src|generated)/instantiations/(inst_[^/\s]+)\.cpp\.o$"
)

rows = []
starts = []
ends = []
for line in log_path.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#"):
        continue
    cols = line.split("\t")
    if len(cols) < 5:
        continue
    try:
        start_ms = int(cols[0])
        end_ms = int(cols[1])
    except ValueError:
        continue
    output_path = cols[3]
    match = pattern.search(output_path)
    if not match:
        continue
    unit = match.group(1)  # inst_<name>
    shard_name = unit[len("inst_") :]
    strategy = "single_tu"
    if shard_name.endswith("_core") or shard_name.endswith("_search"):
        strategy = "core_and_search"
    wall_clock_ms = end_ms - start_ms
    rows.append((shard_name, strategy, wall_clock_ms, start_ms, end_ms))
    starts.append(start_ms)
    ends.append(end_ms)

if not rows:
    print("error: no instantiation object entries found in .ninja_log", file=sys.stderr)
    sys.exit(1)

rows.sort(key=lambda row: row[0])
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["shard_name", "strategy", "wall_clock_ms"])
    for shard_name, strategy, wall_clock_ms, _, _ in rows:
        writer.writerow([shard_name, strategy, wall_clock_ms])

total_wall_clock_ms = max(ends) - min(starts)
print(f"wrote {len(rows)} shard rows to {out_path}")
print(f"total_wall_clock_ms={total_wall_clock_ms}")
PY
