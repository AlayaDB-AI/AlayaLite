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


def run_command(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command and return the exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
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

    print(f"Platform: {platform.system()} (host: {platform.machine()}, target: {get_target_arch()})")
    print(f"Project directory: {project_dir}")
    print(f"Host settings: os={host_os}, arch={host_arch}")
    if host_profile_path is not None:
        if not host_profile_path.exists():
            print(f"Error: Conan profile not found: {host_profile_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Using host Conan profile override: {host_profile_path}")
    else:
        print("Using Conan detected default profile with dynamic host overrides")
    print(f"Build type: {args.build_type}")

    # Detect default profile (for build profile)
    print("\nDetecting default Conan profile...")
    ret = run_command(["uvx", "conan", "profile", "detect", "--force"])
    if ret != 0:
        print(f"Error: conan profile detect failed with exit code {ret}", file=sys.stderr)
        sys.exit(ret)

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
        f"build_type={args.build_type}",
        "-s:b",
        f"build_type={args.build_type}",
        "--build=missing",
    ]

    sdk_path = get_apple_sdk_path()
    if sdk_path is not None:
        cmd.extend(["-c:h", f"tools.apple:sdk_path={sdk_path}"])

    ret = run_command(cmd, cwd=project_dir)

    if ret != 0:
        print(f"Error: conan install failed with exit code {ret}", file=sys.stderr)
        sys.exit(ret)

    print("\nConan install completed successfully!")


if __name__ == "__main__":
    main()
