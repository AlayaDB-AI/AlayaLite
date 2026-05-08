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

"""Static checks for GitHub Actions cache ownership."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
CONAN_CACHE_ACTION = ROOT / ".github" / "actions" / "cache-restore" / "action.yaml"
CMAKE_LISTS = ROOT / "CMakeLists.txt"


def _yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _steps(workflow_name: str, job_name: str) -> list[dict]:
    workflow = _yaml(WORKFLOWS / workflow_name)
    return workflow["jobs"][job_name]["steps"]


def _uses(steps: list[dict], action: str) -> list[dict]:
    return [step for step in steps if step.get("uses") == action]


def _named_step(steps: list[dict], name: str) -> dict:
    for step in steps:
        if step.get("name") == name:
            return step
    raise AssertionError(f"step not found: {name}")


def test_python_only_jobs_do_not_use_conan_cache_action() -> None:
    assert not _uses(_steps("precommit-checker.yaml", "pre-commit"), "./.github/actions/cache-restore")
    assert not _uses(_steps("code-checker.yaml", "py-lint-check"), "./.github/actions/cache-restore")


def test_precommit_has_dedicated_hook_environment_cache() -> None:
    steps = _steps("precommit-checker.yaml", "pre-commit")
    cache_steps = _uses(steps, "actions/cache@v5")

    assert any(step["with"]["path"] == "~/.cache/pre-commit" for step in cache_steps)
    assert any("hashFiles('.pre-commit-config.yaml')" in step["with"]["key"] for step in cache_steps)


def test_setup_uv_cache_keys_are_driven_by_uv_lock_only() -> None:
    setup_steps = [
        step
        for workflow_name in ("precommit-checker.yaml", "code-checker.yaml", "codecov.yaml")
        for workflow in [_yaml(WORKFLOWS / workflow_name)]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("uses", "").startswith("astral-sh/setup-uv@")
    ]

    assert setup_steps
    assert all(step.get("with", {}).get("enable-cache") is True for step in setup_steps)
    assert all(step.get("with", {}).get("cache-dependency-glob") == "uv.lock" for step in setup_steps)


def test_conan_cache_action_uses_explicit_restore_and_save() -> None:
    action = _yaml(CONAN_CACHE_ACTION)
    uses_values = [step.get("uses") for step in action["runs"]["steps"]]

    assert "actions/cache/restore@v5" in uses_values
    assert "actions/cache/save@v5" in uses_values
    assert "actions/cache@v4" not in uses_values
    assert "actions/cache/restore@v4" not in uses_values
    assert "actions/cache/save@v4" not in uses_values


def test_conan_cache_key_includes_install_script_and_target_arch() -> None:
    action = _yaml(CONAN_CACHE_ACTION)
    restore_step = next(step for step in action["runs"]["steps"] if step.get("uses") == "actions/cache/restore@v5")
    key = restore_step["with"]["key"]

    assert "${{ inputs.target-arch }}" in key
    assert "scripts/conan_build/conan_install.py" in key
    assert key.startswith("conan-v2-")


def test_workflow_conan_cache_calls_are_arch_scoped_and_nonfatal_on_save_race() -> None:
    conan_steps = [
        step
        for workflow_name in ("code-checker.yaml", "codecov.yaml", "cibuildwheel.yaml")
        for workflow in [_yaml(WORKFLOWS / workflow_name)]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("uses") == "./.github/actions/cache-restore"
    ]

    assert conan_steps
    assert all("target-arch" in step.get("with", {}) for step in conan_steps)
    assert all(
        step.get("continue-on-error") is True for step in conan_steps if step.get("with", {}).get("mode") == "save"
    )


def test_cpp_heavy_jobs_enable_ccache() -> None:
    unit_steps = _steps("code-checker.yaml", "py-unit-test")
    cpp_steps = _steps("codecov.yaml", "codecov-cpp")

    assert _uses(unit_steps, "hendrikmuhs/ccache-action@v1")
    assert _uses(cpp_steps, "hendrikmuhs/ccache-action@v1")
    assert "CMAKE_CXX_COMPILER_LAUNCHER=ccache" in _named_step(unit_steps, "Build Python test environment")["run"]
    assert "CMAKE_CXX_COMPILER_LAUNCHER=ccache" in _named_step(cpp_steps, "Run c++ code coverage")["run"]


def test_ccache_builds_disable_native_arch() -> None:
    """Do not cache -march=native objects across heterogeneous GitHub runners."""

    unit_steps = _steps("code-checker.yaml", "py-unit-test")
    coverage_steps = _steps("codecov.yaml", "codecov-python")
    wheel_env = _yaml(WORKFLOWS / "cibuildwheel.yaml")["jobs"]["build_wheels"]["env"]
    codecov_script = (ROOT / "scripts" / "ci" / "codecov" / "gnu_codecoverage.sh").read_text(encoding="utf-8")

    assert "-DALAYA_NATIVE_ARCH=OFF" in _named_step(unit_steps, "Build Python test environment")["run"]
    assert "-DALAYA_NATIVE_ARCH=OFF" in _named_step(coverage_steps, "Build Python coverage environment")["run"]
    assert "-DALAYA_NATIVE_ARCH=OFF" in wheel_env["CMAKE_ARGS"]
    assert "-DALAYA_NATIVE_ARCH=OFF" in codecov_script


def test_cibuildwheel_builds_portable_package_targets() -> None:
    wheel_env = _yaml(WORKFLOWS / "cibuildwheel.yaml")["jobs"]["build_wheels"]["env"]
    cmake_args = wheel_env["CMAKE_ARGS"]

    assert "-DALAYA_NATIVE_ARCH=OFF" in cmake_args
    assert "-DBUILD_TOOLS=OFF" in cmake_args


def test_ccache_keys_are_versioned_for_portable_isa_reset() -> None:
    ccache_steps = [
        _uses(_steps("code-checker.yaml", "py-unit-test"), "hendrikmuhs/ccache-action@v1")[0],
        _uses(_steps("codecov.yaml", "codecov-python"), "hendrikmuhs/ccache-action@v1")[0],
        _uses(_steps("codecov.yaml", "codecov-cpp"), "hendrikmuhs/ccache-action@v1")[0],
    ]

    assert all("portable-v2" in step["with"]["key"] for step in ccache_steps)
    assert all("portable-v2" in step["with"]["restore-keys"] for step in ccache_steps)


def test_cmake_defaults_to_portable_native_arch_and_guards_packages() -> None:
    cmake_text = CMAKE_LISTS.read_text(encoding="utf-8")

    assert 'option(ALAYA_NATIVE_ARCH "Compile with -march=native for host-specific builds" OFF)' in cmake_text
    assert "option(ALAYA_ALLOW_NATIVE_PACKAGE" in cmake_text
    assert "if(BUILD_PYTHON" in cmake_text
    assert "AND ALAYA_NATIVE_ARCH" in cmake_text
    assert "AND NOT ALAYA_ALLOW_NATIVE_PACKAGE" in cmake_text


def test_portable_cmake_uses_avx2_baseline_without_native_or_avx512() -> None:
    cmake_text = CMAKE_LISTS.read_text(encoding="utf-8")

    assert "list(APPEND ALAYA_SIMD_COMPILE_OPTIONS -mavx2 -mfma)" in cmake_text
    assert "list(APPEND ALAYA_SIMD_COMPILE_OPTIONS -mavx512" not in cmake_text
