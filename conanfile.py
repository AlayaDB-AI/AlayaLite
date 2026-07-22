# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

import os
import re

from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy


def _read_version():
    pyproject = os.path.join(os.path.dirname(__file__), "pyproject.toml")
    with open(pyproject, encoding="utf-8") as f:
        m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
    return m.group(1) if m else "0.0.0"


class AlayaLiteConan(ConanFile):
    name = "AlayaLite"
    version = _read_version()
    settings = "os", "compiler", "build_type", "arch"
    package_type = "header-library"
    exports = "pyproject.toml"
    exports_sources = "include/*"
    platform_tool_requires = "cmake/3.23.5"  # cmake version

    def layout(self):
        """
        Let Conan automatically handle the build directory structure
        Resolve output conflicts between different compilers across platforms
        """
        cmake_layout(self)

    def requirements(self):
        self.requires("gtest/1.16.0")
        self.requires("concurrentqueue/1.0.4")
        self.requires("pybind11/2.13.6")
        self.requires("spdlog/1.14.0")
        self.requires("eigen/3.4.0")
        self.requires("lz4/1.9.4")
        self.requires("zstd/1.5.6")
        self.requires("rocksdb/10.5.1")
        # DiskANN uses libcoro's portable task/thread-pool primitives on every
        # platform; only its io_uring accelerator is Linux-specific.
        self.requires("libcoro/0.14.1")
        if self.settings.os == "Linux":
            self.requires("liburing/2.13")

        # OpenMP support
        if self.settings.os == "Linux":
            if self.settings.compiler in ["clang", "apple-clang"]:
                self.requires("llvm-openmp/17.0.6")
        # GCC: libgomp is system-provided.
        # macOS wheel builds use Homebrew libomp from cibuildwheel before-all.
        # Windows MSVC: LASER only uses OpenMP 2.0 constructs (parallel for /
        # critical / schedule), so the toolchain-default /openmp + vcomp140.dll
        # runtime is sufficient. find_package(OpenMP) resolves to that path
        # automatically — no Conan / vcpkg OpenMP dependency required.

    def configure(self):
        # Static link all dependencies
        self.options["*"].shared = False
        self.options["*"].fPIC = True

        # Use header-only spdlog to avoid ABI compatibility issues on Windows
        self.options["spdlog"].header_only = True

        # Enable compression libraries for RocksDB
        self.options["rocksdb"].with_lz4 = True
        self.options["rocksdb"].with_zstd = True

        # The project uses libcoro task/mutex primitives, not its networking/TLS
        # layer. ConanCenter's libcoro networking option pulls in epoll-only
        # sources that do not build on macOS.
        self.options["libcoro"].with_networking = False
        self.options["libcoro"].with_ssl = False

    def generate(self):
        tc = CMakeToolchain(self)
        # Dependencies flow through the CMake dependency provider (cmake/AlayaConan.cmake), which only
        # consumes the CMakeDeps output; a generated CMakeUserPresets.json would go stale and break
        # `cmake --preset` after `make clean`. The toolchain file is still emitted for manual
        # (-DALAYA_AUTO_CONAN=OFF) workflows.
        tc.user_presets_path = None
        tc.generate()
        cmake = CMakeDeps(self)
        cmake.generate()

    def package(self):
        copy(
            self,
            "*.h",
            src=os.path.join(self.export_sources_folder, "include"),
            dst=os.path.join(self.package_folder, "include"),
        )
        copy(
            self,
            "*.hpp",
            src=os.path.join(self.export_sources_folder, "include"),
            dst=os.path.join(self.package_folder, "include"),
        )

    def package_info(self):
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
