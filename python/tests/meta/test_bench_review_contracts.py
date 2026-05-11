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

"""Static review-contract tests for benchmark smoke tests."""

from __future__ import annotations

import ast
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCH_TESTS = ROOT / "python" / "tests" / "bench"


def _read_bench_file(name: str) -> str:
    return (BENCH_TESTS / name).read_text(encoding="utf-8")


def test_conftest_short_circuits_windows_before_extension_import() -> None:
    text = _read_bench_file("conftest.py")

    assert text.index('if sys.platform == "win32":') < text.index("pytest.importorskip(")


def test_conftest_ignores_bench_tests_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    namespace = runpy.run_path(str(BENCH_TESTS / "conftest.py"))

    assert namespace["collect_ignore_glob"] == ["*.py"]


def test_disk_engine_synth_schema_pins_metric_argument() -> None:
    text = _read_bench_file("test_disk_engine_synth_schema.py")

    assert '"--metric"' in text
    assert '"L2"' in text


def test_distance_order_cases_include_query_vectors_in_parametrize() -> None:
    text = _read_bench_file("test_datasets.py")

    assert '"metric, vecs, q"' in text
    assert "if metric" not in text


def test_dataset_tests_docstring_matches_bench_extension_guard() -> None:
    module = ast.parse(_read_bench_file("test_datasets.py"))
    docstring = ast.get_docstring(module)

    assert docstring is not None
    assert "no C++ extension needed" not in docstring
