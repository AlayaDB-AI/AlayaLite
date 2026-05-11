# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for benchmark compatibility wrappers."""

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_module(path: Path):
    # `python/benchmarks/` is not on sys.path as a package, so spec-load
    # the wrapper script directly rather than relying on `import`.
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_disk_laser_wrapper_forwards_legacy_input_flags(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    module = _load_module(repo_root / "python" / "benchmarks" / "disk_laser_smoke.py")
    captured: dict = {}

    def fake_run(cmd, check=False):
        captured["cmd"] = cmd
        captured["check"] = check
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "disk_laser_smoke.py",
            "--src-dir",
            "/tmp/laser",
            "--vectors",
            "/tmp/base.fbin",
            "--queries-path",
            "/tmp/query.fbin",
            "--ground-truth",
            "/tmp/gt.ibin",
        ],
    )

    assert module.main() == 0
    cmd = captured["cmd"]
    assert "--dataset" in cmd and cmd[cmd.index("--dataset") + 1] == "laser_files"
    assert "--vectors" in cmd and cmd[cmd.index("--vectors") + 1] == "/tmp/base.fbin"
    assert "--queries-path" in cmd and cmd[cmd.index("--queries-path") + 1] == "/tmp/query.fbin"
    assert "--ground-truth" in cmd and cmd[cmd.index("--ground-truth") + 1] == "/tmp/gt.ibin"
