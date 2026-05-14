#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Compare the current _alayalitepy.so ABI against a checked-in baseline.
# Guards against unintentional ABI drift. Internally calls snapshot.sh to
# dump the current .so symbols and diffs the filtered set against baseline.
#
# When to run:
# - Locally before pushing a release tag
# - When investigating "old client crashes against new .so" reports
# - Optionally in CI after build (not currently wired)
#
# Usage:
#     bash scripts/abi/diff.sh <baseline.txt> <shared_object.so>
#
# Filter:
#     ALAYA_ABI_SYMBOL_REGEX env var narrows the comparison
#     (default: "alaya|PyInit_" — drops stdlib / zstd / runtime noise).
#     Example: ALAYA_ABI_SYMBOL_REGEX="alaya::PyIndex" bash diff.sh ...
#
# Exit code:
#     0  ABI matches baseline
#     1  ABI drift detected (unified diff printed)
#     2  bad usage / missing args

set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 <baseline.txt> <shared_object.so>" >&2
  echo "env:   ALAYA_ABI_SYMBOL_REGEX (default: alaya|PyInit_)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_PATH="$1"
SO_PATH="$2"
SYMBOL_REGEX="${ALAYA_ABI_SYMBOL_REGEX:-alaya|PyInit_}"

TMP_SNAPSHOT="$(mktemp)"
TMP_BASELINE_FILTERED="$(mktemp)"
TMP_SNAPSHOT_FILTERED="$(mktemp)"
trap 'rm -f "${TMP_SNAPSHOT}" "${TMP_BASELINE_FILTERED}" "${TMP_SNAPSHOT_FILTERED}"' EXIT

"${SCRIPT_DIR}/snapshot.sh" "${TMP_SNAPSHOT}" "${SO_PATH}" >/dev/null

awk -v re="${SYMBOL_REGEX}" '$0 ~ re {print}' "${BASELINE_PATH}" | sort -u >"${TMP_BASELINE_FILTERED}"
awk -v re="${SYMBOL_REGEX}" '$0 ~ re {print}' "${TMP_SNAPSHOT}" | sort -u >"${TMP_SNAPSHOT_FILTERED}"

if diff -u "${TMP_BASELINE_FILTERED}" "${TMP_SNAPSHOT_FILTERED}"; then
  echo "ABI diff: no changes"
  exit 0
fi

echo "ABI diff: detected changes (filter: ${SYMBOL_REGEX})" >&2
exit 1
