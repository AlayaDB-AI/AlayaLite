#!/usr/bin/env python3
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


def get_profile_path(script_dir: Path) -> Path:
    """
    Select the appropriate Conan profile based on platform and architecture.

    Profile selection logic (matches cmake/ConanSetup.cmake):
    - Windows: conan_profile_win.x86_64
    - macOS aarch64/arm64: conan_profile_mac.aarch64
    - macOS x86_64: conan_profile_mac.x86_64
    - Linux aarch64/arm64: conan_profile.aarch64
    - Linux x86_64: conan_profile.x86_64
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

    # Get the appropriate profile
    profile_path = get_profile_path(script_dir)

    if not profile_path.exists():
        print(f"Error: Conan profile not found: {profile_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Platform: {platform.system()} (host: {platform.machine()}, target: {get_target_arch()})")
    print(f"Project directory: {project_dir}")
    print(f"Using Conan profile: {profile_path}")
    print(f"Build type: {args.build_type}")

    # Detect default profile (for build profile)
    print("\nDetecting default Conan profile...")
    run_command(["uvx", "conan", "profile", "detect", "--force"])

    # Run conan install
    print("\nRunning conan install...")
    cmd = [
        "uvx",
        "conan",
        "install",
        str(project_dir),
        "-pr:h",
        str(profile_path),
        "-pr:b",
        "default",
        "-s",
        f"build_type={args.build_type}",
        "--build=missing",
    ]

    ret = run_command(cmd, cwd=project_dir)

    if ret != 0:
        print(f"Error: conan install failed with exit code {ret}", file=sys.stderr)
        sys.exit(ret)

    print("\nConan install completed successfully!")


if __name__ == "__main__":
    main()
