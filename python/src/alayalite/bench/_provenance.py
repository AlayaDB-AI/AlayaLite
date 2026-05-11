# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Provenance collection for DiskCollection benchmark reports."""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Sequence, Union

import numpy as np

# Per `disk-collection-benchmark-suite/design.md` D8, the compiler flag
# string is a v1 hardcoded literal (NOT pulled from CMake or wheel
# metadata). `compiler_flags_source="hardcoded_v1"` is the field consumers
# MUST inspect before trusting the literal.
#
# This will silently disagree with reality if any of the following hold:
#  - the wheel was rebuilt with a non-default `CMAKE_CXX_FLAGS` override
#  - someone exported `CXXFLAGS=...` to their shell before pip / uv build
#  - a distribution repackages the project with custom build flags
#  - a future CMake change alters the default flag set
# Future change `disk-collection-benchmark-flags-from-cmake` is expected to
# replace this with a generated source-of-truth pulled from the wheel.
COMPILER_FLAGS = "-Ofast -DNDEBUG -march=native"
COMPILER_FLAGS_SOURCE = "hardcoded_v1"
_SIMD_FILTER = {"sse4_1", "sse4_2", "avx", "avx2", "avx512f", "avx512bw", "avx512vl", "avx512dq"}


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def _git_state() -> tuple[str, Union[bool, str]]:
    rev = _run_git(["git", "rev-parse", "HEAD"])
    if rev.returncode != 0:
        return "unknown", "unknown"

    dirty = _run_git(["git", "diff", "--quiet", "HEAD"])
    return rev.stdout.strip(), dirty.returncode != 0


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def parse_cpuinfo(text: str) -> tuple[str, list[str]]:
    cpu_model = ""
    flags: set[str] = set()
    for line in text.splitlines():
        if not cpu_model and line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
        if line.startswith("flags"):
            flags.update(flag for flag in line.split(":", 1)[1].split() if flag in _SIMD_FILTER)
    return cpu_model, sorted(flags)


def parse_mem_total_kb(text: str) -> int:
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return 0
    return 0


def collect_provenance(seed: int, dataset_sha16: str, harness_argv: Sequence[str]) -> dict:
    if len(dataset_sha16) != 16 or any(ch not in "0123456789abcdef" for ch in dataset_sha16.lower()):
        raise ValueError("dataset_sha16 must be exactly 16 hex characters")

    git_commit_sha, git_dirty = _git_state()
    cpu_model, simd_flags = parse_cpuinfo(_read_text("/proc/cpuinfo"))

    try:
        alayalite_version = metadata.version("alayalite")
    except metadata.PackageNotFoundError:
        alayalite_version = "unknown"

    uname = platform.uname()
    return {
        "git_commit_sha": git_commit_sha,
        "git_dirty": git_dirty,
        "uname": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "cpu_simd_flags": simd_flags,
        "mem_total_kb": parse_mem_total_kb(_read_text("/proc/meminfo")),
        "compiler_flags": COMPILER_FLAGS,
        "compiler_flags_source": COMPILER_FLAGS_SOURCE,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "alayalite_version": alayalite_version,
        "dataset_sha256_prefix": dataset_sha16,
        "dataset_sha256_prefix_source": "dataset",
        "seed": int(seed),
        "harness_argv": list(harness_argv),
        "timestamp_iso8601": datetime.now(timezone.utc).isoformat(),
    }
