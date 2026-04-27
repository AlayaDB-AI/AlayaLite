#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Conan dependency installation script.

This script handles Conan profile detection and dependency installation
across different platforms. It mirrors the logic in cmake/ConanSetup.cmake
to ensure consistent behavior between local builds and CI.

Usage:
    python scripts/conan_build/conan_install.py [--build-type TYPE] [--project-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path


def get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def get_target_arch() -> str:
    """
    Detect target architecture from cibuildwheel env vars or platform.machine().
    """
    system = platform.system()

    # Check CIBW_ARCHS_* (set by GitHub workflow)
    if system == "Darwin":
        cibw_archs = os.environ.get("CIBW_ARCHS_MACOS", "")
    elif system == "Linux":
        cibw_archs = os.environ.get("CIBW_ARCHS_LINUX", "")
    elif system == "Windows":
        cibw_archs = os.environ.get("CIBW_ARCHS_WINDOWS", "")
    else:
        cibw_archs = ""

    if cibw_archs:
        if "arm64" in cibw_archs or "aarch64" in cibw_archs or "ARM64" in cibw_archs:
            return "arm64"
        if "x86_64" in cibw_archs or "AMD64" in cibw_archs:
            return "x86_64"

    return platform.machine().lower()


def get_static_profile_path(script_dir: Path) -> Path:
    """
    Select the bundled Conan profile based on platform and architecture.

    Bundled profiles remain available as an override, but the default flow now
    prefers Conan's detected profile so compiler versions do not need to be
    hard-coded per platform.
    """
    system = platform.system()
    machine = get_target_arch()

    if system == "Windows":
        profile_name = "conan_profile_win.x86_64"
    elif system == "Darwin":
        if machine in ("arm64", "aarch64"):
            profile_name = "conan_profile_mac.aarch64"
        else:
            profile_name = "conan_profile_mac.x86_64"
    elif system == "Linux":
        if machine in ("aarch64", "arm64"):
            profile_name = "conan_profile.aarch64"
        else:
            profile_name = "conan_profile.x86_64"
    else:
        print(f"Error: Unsupported platform: {system}", file=sys.stderr)
        sys.exit(1)

    return script_dir / profile_name


def get_host_profile_path(script_dir: Path) -> Path | None:
    """
    Resolve an optional host profile override.

    Priority:
    1. `ALAYA_CONAN_PROFILE`
    2. bundled static profile when `ALAYA_USE_STATIC_CONAN_PROFILE=1`
    3. no override -> use Conan's detected default profile
    """
    profile_override = os.environ.get("ALAYA_CONAN_PROFILE")
    if profile_override:
        return Path(profile_override).expanduser().resolve()

    if os.environ.get("ALAYA_USE_STATIC_CONAN_PROFILE") == "1":
        return get_static_profile_path(script_dir)

    return None


def get_conan_os() -> str:
    """Map Python platform names to Conan OS setting names."""
    system = platform.system()
    if system == "Darwin":
        return "Macos"
    if system in {"Linux", "Windows"}:
        return system
    print(f"Error: Unsupported platform: {system}", file=sys.stderr)
    sys.exit(1)


def get_conan_arch() -> str:
    """Map host/target architecture names to Conan arch setting names."""
    machine = get_target_arch()
    arch_map = {
        "aarch64": "armv8",
        "arm64": "armv8",
        "amd64": "x86_64",
        "x86_64": "x86_64",
    }
    if machine not in arch_map:
        print(f"Error: Unsupported architecture: {machine}", file=sys.stderr)
        sys.exit(1)
    return arch_map[machine]


def get_conan_cppstd() -> str:
    """
    Return the C++ standard required by the project for Conan resolution.

    The default must stay aligned with `CMAKE_CXX_STANDARD` in CMakeLists.txt.
    Allow an environment override for CI/debugging edge cases.
    """
    return os.environ.get("ALAYA_CONAN_CPPSTD", "20")


def get_apple_sdk_path() -> str | None:
    """Best-effort discovery of the active Apple SDK path."""
    if platform.system() != "Darwin":
        return None

    result = subprocess.run(
        ["xcrun", "--show-sdk-path"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    sdk_path = result.stdout.strip()
    return sdk_path or None


def get_apple_clang_version() -> str | None:
    """Best-effort discovery of the active Apple Clang major version."""
    if platform.system() != "Darwin":
        return None

    for cmd in (["xcrun", "clang", "--version"], ["clang", "--version"]):
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue

        match = re.search(r"Apple clang version\s+(\d+)", result.stdout)
        if match is not None:
            return match.group(1)

    return None


def get_default_profile_path(env: dict[str, str]) -> Path:
    """Resolve Conan's default profile path for the current environment."""
    conan_home = env.get("CONAN_HOME")
    if conan_home:
        return Path(conan_home).expanduser().resolve() / "profiles" / "default"
    return Path.home() / ".conan2" / "profiles" / "default"


def read_profile_settings(profile_path: Path) -> dict[str, str]:
    """Read the `[settings]` section from a Conan profile."""
    if not profile_path.exists():
        return {}

    settings: dict[str, str] = {}
    in_settings = False

    for raw_line in profile_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            in_settings = line == "[settings]"
            continue

        if not in_settings or "=" not in line:
            continue

        key, value = line.split("=", 1)
        settings[key.strip()] = value.strip()

    return settings


def get_default_profile_compiler_overrides(env: dict[str, str]) -> list[tuple[str, str]]:
    """
    Fill in missing compiler settings when Conan detects an incomplete default profile.

    This mainly protects macOS wheel/editable builds where Python tooling may
    expose `CC=gcc` / `CXX=g++`, which are Apple Clang driver aliases and can
    confuse Conan's profile detection.
    """
    if platform.system() != "Darwin":
        return []

    profile_settings = read_profile_settings(get_default_profile_path(env))
    overrides: list[tuple[str, str]] = []

    compiler = profile_settings.get("compiler")
    if not compiler:
        overrides.append(("compiler", "apple-clang"))
        compiler = "apple-clang"

    if compiler != "apple-clang":
        return overrides

    if not profile_settings.get("compiler.version"):
        compiler_version = get_apple_clang_version()
        if compiler_version is not None:
            overrides.append(("compiler.version", compiler_version))

    if not profile_settings.get("compiler.libcxx"):
        overrides.append(("compiler.libcxx", "libc++"))

    return overrides


def get_conan_env() -> dict[str, str]:
    """
    Normalize compiler-related environment variables for Conan invocations.

    Some Python build environments export `CC`/`CXX` as full compiler commands
    such as `cc -pthread`. Conan's `profile detect` expects the variable value
    to be the compiler driver name, so strip extra flags and move them to the
    standard flags variables.
    """
    env = os.environ.copy()

    def normalize_compiler_driver(compiler_cmd: str, *, compiler_var: str) -> tuple[str, str]:
        parts = shlex.split(compiler_cmd)
        if not parts:
            return compiler_cmd, ""

        compiler_name = os.path.basename(parts[0])

        if platform.system() == "Darwin":
            alias_map = {
                "CC": {
                    "cc": "clang",
                    "gcc": "clang",
                },
                "CXX": {
                    "c++": "clang++",
                    "g++": "clang++",
                },
            }
            compiler_name = alias_map.get(compiler_var, {}).get(compiler_name, compiler_name)

        return compiler_name, " ".join(parts[1:])

    for compiler_var, flags_var in (("CC", "CFLAGS"), ("CXX", "CXXFLAGS")):
        compiler_cmd = env.get(compiler_var)
        if not compiler_cmd:
            continue

        normalized_compiler, extra_flags = normalize_compiler_driver(compiler_cmd, compiler_var=compiler_var)
        env[compiler_var] = normalized_compiler
        if extra_flags:
            existing_flags = env.get(flags_var, "")
            env[flags_var] = f"{extra_flags} {existing_flags}".strip()

        if compiler_cmd != normalized_compiler:
            print(f"Normalized {compiler_var} for Conan: {compiler_cmd!r} -> {normalized_compiler!r}")

    return env


def run_command(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    """Run a command and return the exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Install Conan dependencies with platform-specific profile")
    parser.add_argument(
        "--build-type",
        default="Release",
        choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
        help="CMake build type (default: Release)",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Project root directory (default: auto-detect from script location)",
    )
    args = parser.parse_args()

    script_dir = get_script_dir()

    # Determine project directory
    if args.project_dir:
        project_dir = args.project_dir.resolve()
    else:
        # Script is in scripts/conan_build/, project root is two levels up
        project_dir = script_dir.parent.parent

    host_profile_path = get_host_profile_path(script_dir)
    host_os = get_conan_os()
    host_arch = get_conan_arch()
    host_cppstd = get_conan_cppstd()

    print(f"Platform: {platform.system()} (host: {platform.machine()}, target: {get_target_arch()})")
    print(f"Project directory: {project_dir}")
    print(f"Host settings: os={host_os}, arch={host_arch}, cppstd={host_cppstd}")
    if host_profile_path is not None:
        if not host_profile_path.exists():
            print(f"Error: Conan profile not found: {host_profile_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Using host Conan profile override: {host_profile_path}")
    else:
        print("Using Conan detected default profile with dynamic host overrides")
    print(f"Build type: {args.build_type}")

    conan_env = get_conan_env()

    # Detect default profile (for build profile)
    print("\nDetecting default Conan profile...")
    ret = run_command(["uvx", "conan", "profile", "detect", "--force"], env=conan_env)
    if ret != 0:
        print(f"Error: conan profile detect failed with exit code {ret}", file=sys.stderr)
        sys.exit(ret)

    default_profile_compiler_overrides = get_default_profile_compiler_overrides(conan_env)
    if default_profile_compiler_overrides:
        print("Detected incomplete Conan default profile; applying explicit macOS compiler overrides")

    # Run conan install
    print("\nRunning conan install...")
    cmd = [
        "uvx",
        "conan",
        "install",
        str(project_dir),
        "-pr:h",
        str(host_profile_path) if host_profile_path is not None else "default",
        "-pr:b",
        "default",
        "-s:h",
        f"os={host_os}",
        "-s:h",
        f"arch={host_arch}",
        "-s:h",
        f"compiler.cppstd={host_cppstd}",
        "-s:h",
        f"build_type={args.build_type}",
        "-s:b",
        f"build_type={args.build_type}",
        "--build=missing",
    ]

    if host_profile_path is None:
        for key, value in default_profile_compiler_overrides:
            cmd.extend(["-s:h", f"{key}={value}"])

    for key, value in default_profile_compiler_overrides:
        cmd.extend(["-s:b", f"{key}={value}"])

    sdk_path = get_apple_sdk_path()
    if sdk_path is not None:
        cmd.extend(["-c:h", f"tools.apple:sdk_path={sdk_path}"])

    ret = run_command(cmd, cwd=project_dir, env=conan_env)

    if ret != 0:
        print(f"Error: conan install failed with exit code {ret}", file=sys.stderr)
        sys.exit(ret)

    print("\nConan install completed successfully!")


if __name__ == "__main__":
    main()
