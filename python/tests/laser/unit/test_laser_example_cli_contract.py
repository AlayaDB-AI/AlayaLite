# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Contract tests for examples/laser/main.py CLI config loading."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


def _load_example_main():
    repo_root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "laser_example_main",
        repo_root / "examples" / "laser" / "main.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        topk=None,
        threads=None,
        beam_width=None,
        dram_budget=None,
        efs=None,
        ep_num=None,
        degree=None,
        main_dim=None,
        build_threads=None,
        ef_indexing=None,
        warmup=None,
        runs=None,
    )


def _write_config(path: Path, extra_top_level: str = "") -> None:
    path.write_text(
        f"""
{extra_top_level}
seed = 99

[dataset]
name = "test"
metric = "l2"
degree = 64
main_dimension = 128

[paths]
base = "/tmp/base.fbin"
query = "/tmp/query.fbin"
gt = "/tmp/gt.ibin"
output = "/tmp/out"

[build_vamana]
L = 200
alpha = 1.2
dram_budget_gb = 1.0

[build]
build_threads = 1
ef_indexing = 200

[search]
topk = 10
threads = 1
beam_width = 16
dram_budget = 1.0
ep_num = 300
warmup = 0
runs = 1
efs = [100]
""".lstrip(),
        encoding="utf-8",
    )


def test_load_config_rejects_legacy_alignment_step_fields(tmp_path: Path) -> None:
    module = _load_example_main()
    cfg = tmp_path / "legacy.toml"
    _write_config(cfg, "pca_seed = 42\nmedoid_seed = 42\nrotator_seed = 42\ndump_rotator = true")

    with pytest.raises(ValueError, match="legacy.*pca_seed.*seed"):
        module.load_config(cfg, _args())


def test_load_config_returns_unified_fit_contract_only(tmp_path: Path) -> None:
    module = _load_example_main()
    cfg = tmp_path / "unified.toml"
    _write_config(cfg)

    loaded = module.load_config(cfg, _args())

    assert loaded["seed"] == 99
    assert loaded["build_vamana_L"] == 200
    assert loaded["build_vamana_alpha"] == 1.2
    assert loaded["build_vamana_dram_budget_gb"] == 1.0
    assert "vamana" not in loaded
    assert "pca_seed" not in loaded
    assert "medoid_seed" not in loaded
    assert "rotator_seed" not in loaded
    assert "dump_rotator" not in loaded
