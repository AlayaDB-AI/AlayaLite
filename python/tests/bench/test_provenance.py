# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for benchmark provenance helpers."""

import subprocess

from alayalite.bench import _provenance
from alayalite.bench._metrics import render_summary_json


def test_parse_cpuinfo_filters_sorted_simd_flags():
    text = """
model name  : Example CPU
flags       : avx2 sse4_2 fma avx512f avx sse4_1 avx2
"""
    model, flags = _provenance.parse_cpuinfo(text)
    assert model == "Example CPU"
    assert flags == ["avx", "avx2", "avx512f", "sse4_1", "sse4_2"]


def test_collect_provenance_fields_and_git_fallback(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 1, "", "no git")

    def fake_read_text(path):
        if path == "/proc/cpuinfo":
            return "model name\t: CPU\nflags\t: avx avx512bw sse4_1 nope\n"
        if path == "/proc/meminfo":
            return "MemTotal:       12345 kB\n"
        return ""

    monkeypatch.setattr(_provenance.subprocess, "run", fake_run)
    monkeypatch.setattr(_provenance, "_read_text", fake_read_text)

    provenance = _provenance.collect_provenance(42, "0123456789abcdef", ["bench", "--x"])
    assert provenance["git_commit_sha"] == "unknown"
    assert provenance["git_dirty"] == "unknown"
    assert provenance["cpu_model"] == "CPU"
    assert provenance["cpu_simd_flags"] == ["avx", "avx512bw", "sse4_1"]
    assert provenance["mem_total_kb"] == 12345
    assert provenance["compiler_flags"] == "-Ofast -DNDEBUG -march=native"
    assert provenance["compiler_flags_source"] == "hardcoded_v1"
    assert provenance["dataset_sha256_prefix"] == "0123456789abcdef"
    assert provenance["seed"] == 42
    assert provenance["harness_argv"] == ["bench", "--x"]


def test_summary_provenance_aggregates_multiple_dataset_hashes():
    base = _provenance.collect_provenance(42, "aaaaaaaaaaaaaaaa", ["bench"])
    other = {**base, "dataset_sha256_prefix": "bbbbbbbbbbbbbbbb"}

    summary = render_summary_json(
        [{"provenance": base}, {"provenance": other}],
        "run",
        base,
    )

    assert summary["provenance"]["dataset_sha256_prefix"] not in {
        "aaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbb",
    }
    assert summary["provenance"]["dataset_sha256_prefix_source"] == "aggregate_multiple_datasets"
    assert summary["provenance"]["dataset_sha256_prefix_members"] == [
        "aaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbb",
    ]
