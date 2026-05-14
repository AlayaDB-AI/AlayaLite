#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Capture the current ABI of _alayalitepy.so as a sorted nm dump. The output
# becomes the new ABI baseline for diff.sh to compare against. Use this only
# when you intentionally change the ABI and want to refresh
# scripts/abi/baseline.txt in the same commit.
#
# When to run:
# - After intentionally adding/removing/renaming a PyIndex method
# - After changing extern template declarations
# - When refreshing the baseline before a release
#
# Usage:
#     bash scripts/abi/snapshot.sh [output_path] [shared_object_path]
#     # default output_path = scripts/abi/baseline.txt
#     # default .so       = first match under build/**/_alayalitepy*.so
#
# Example (refresh baseline after intentional ABI change):
#     bash scripts/abi/snapshot.sh scripts/abi/baseline.txt \
#         build/cp310-cp310-linux_x86_64/python/alayalite/_alayalitepy*.so
#     git add scripts/abi/baseline.txt
#
# Output:
#     Mangled-symbol list (sorted, dedup'd) written to output_path.
#     Stdout: snapshot path, .so path, symbol_count.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTPUT_PATH="${1:-${REPO_ROOT}/scripts/abi/baseline.txt}"
SO_PATH="${2:-}"

if [[ -z "${SO_PATH}" ]]; then
  mapfile -t candidates < <(find "${REPO_ROOT}/build" -type f -name "_alayalitepy*.so" | sort)
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "error: cannot find _alayalitepy shared object under build/" >&2
    exit 1
  fi
  SO_PATH="${candidates[0]}"
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"
nm --dynamic --defined-only "${SO_PATH}" | awk '{print $3}' | sed '/^$/d' | sort -u > "${OUTPUT_PATH}"

echo "snapshot written: ${OUTPUT_PATH}"
echo "shared object: ${SO_PATH}"
echo "symbol_count: $(wc -l < "${OUTPUT_PATH}")"
