# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Static checks for GitHub Actions cache ownership."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
CONAN_CACHE_ACTION = ROOT / ".github" / "actions" / "cache-restore" / "action.yaml"
CMAKE_LISTS = ROOT / "CMakeLists.txt"
PYPROJECT = ROOT / "pyproject.toml"
PRINT_SUMMARY = ROOT / "cmake" / "PrintSummary.cmake"


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
        for workflow_path in WORKFLOWS.glob("*.yaml")
        for workflow in [_yaml(workflow_path)]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("uses") == "./.github/actions/cache-restore"
    ]

    assert conan_steps
    assert all("target-arch" in step.get("with", {}) for step in conan_steps)
    assert all(
        step.get("continue-on-error") is True for step in conan_steps if step.get("with", {}).get("mode") == "save"
    )


def test_coverage_jobs_enable_ccache() -> None:
    python_steps = _steps("codecov.yaml", "codecov-python")
    cpp_steps = _steps("codecov.yaml", "codecov-cpp")

    assert _uses(python_steps, "hendrikmuhs/ccache-action@v1")
    assert _uses(cpp_steps, "hendrikmuhs/ccache-action@v1")
    assert "CMAKE_CXX_COMPILER_LAUNCHER=ccache" in _named_step(python_steps, "Build Python coverage environment")["run"]
    assert "CMAKE_CXX_COMPILER_LAUNCHER=ccache" in _named_step(cpp_steps, "Run c++ code coverage")["run"]


def test_ci_workflow_does_not_run_duplicate_python_unit_job() -> None:
    workflow = _yaml(WORKFLOWS / "code-checker.yaml")

    assert "py-unit-test" not in workflow["jobs"]


def test_codecov_python_replaces_unit_test_gate_without_upload_flakes() -> None:
    coverage_steps = _steps("codecov.yaml", "codecov-python")
    run_step = _named_step(coverage_steps, "Run python code coverage")
    upload_step = _named_step(coverage_steps, "Upload Python coverage to Codecov")

    assert "python_coverage_with_crash_diagnostics.sh" in run_step["run"]
    assert upload_step["continue-on-error"] is True
    assert upload_step["uses"] == "codecov/codecov-action@v5"


def test_codecov_python_build_uses_unit_build_configuration() -> None:
    coverage_build = _named_step(_steps("codecov.yaml", "codecov-python"), "Build Python coverage environment")["run"]

    assert "CMAKE_CXX_COMPILER_LAUNCHER=ccache" in coverage_build
    assert "-DALAYA_NATIVE_ARCH=OFF" in coverage_build
    assert "uv sync --dev --locked" in coverage_build
    assert "CXXFLAGS" not in coverage_build
    assert "install.strip" not in coverage_build


def test_codecov_workflow_keeps_coverage_trigger_scope() -> None:
    workflow = _yaml(WORKFLOWS / "codecov.yaml")
    triggers = workflow.get("on", workflow.get(True))
    coverage_paths = [
        "include/**",
        "tests/**",
        "python/src/**",
        "python/tests/**",
        "app/**",
        "app/tests/**",
        "CMakeLists.txt",
        "Makefile",
        "pyproject.toml",
        "python/CMakeLists.txt",
        "cmake/**",
        "scripts/ci/codecov/**",
        "scripts/conan_build/**",
    ]

    assert triggers is not None
    assert triggers["pull_request"] == {"paths": coverage_paths}
    assert triggers["push"] == {"branches": ["main"], "paths": coverage_paths}


def test_ccache_builds_disable_native_arch() -> None:
    """Do not cache -march=native objects across heterogeneous GitHub runners."""

    coverage_steps = _steps("codecov.yaml", "codecov-python")
    wheel_env = _yaml(WORKFLOWS / "cibuildwheel.yaml")["jobs"]["build_wheels"]["env"]
    codecov_script = (ROOT / "scripts" / "ci" / "codecov" / "gnu_codecoverage.sh").read_text(encoding="utf-8")

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
        _uses(_steps("codecov.yaml", "codecov-python"), "hendrikmuhs/ccache-action@v1")[0],
        _uses(_steps("codecov.yaml", "codecov-cpp"), "hendrikmuhs/ccache-action@v1")[0],
    ]

    assert "portable-v3" in ccache_steps[0]["with"]["key"]
    assert "portable-v2" in ccache_steps[1]["with"]["key"]
    assert all("portable-v" in step["with"]["restore-keys"] for step in ccache_steps)


def test_codecov_python_restores_legacy_unit_test_cache() -> None:
    ccache_step = _uses(_steps("codecov.yaml", "codecov-python"), "hendrikmuhs/ccache-action@v1")[0]

    assert "codecov-python-portable-v3" in ccache_step["with"]["key"]
    assert "py-unit-portable-v2" in ccache_step["with"]["restore-keys"]


def test_cmake_defaults_to_portable_native_arch_and_guards_packages() -> None:
    cmake_text = CMAKE_LISTS.read_text(encoding="utf-8")

    assert 'option(ALAYA_NATIVE_ARCH "Compile with -march=native for host-specific builds" OFF)' in cmake_text
    assert "option(ALAYA_ALLOW_NATIVE_PACKAGE" in cmake_text
    assert "if(BUILD_PYTHON" in cmake_text
    assert "AND ALAYA_NATIVE_ARCH" in cmake_text
    assert "AND NOT ALAYA_ALLOW_NATIVE_PACKAGE" in cmake_text


def test_native_arch_option_name_is_used_consistently() -> None:
    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    summary_text = PRINT_SUMMARY.read_text(encoding="utf-8")

    assert 'ALAYA_NATIVE_ARCH="OFF"' in pyproject_text
    assert "${ALAYA_NATIVE_ARCH}" in summary_text
    assert "ALAYA_ENABLE_NATIVE_OPT" not in pyproject_text
    assert "ALAYA_ENABLE_NATIVE_OPT" not in summary_text


def test_portable_cmake_uses_avx2_baseline_without_native_or_avx512() -> None:
    cmake_text = CMAKE_LISTS.read_text(encoding="utf-8")

    assert "list(APPEND ALAYA_SIMD_COMPILE_OPTIONS -mavx2 -mfma)" in cmake_text
    assert "list(APPEND ALAYA_SIMD_COMPILE_OPTIONS -mavx512" not in cmake_text
