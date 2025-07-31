from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain


class AlayaLiteConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    package_type = "header-library"
    exports_sources = "include/*"
    platform_tool_requires = "cmake/3.23.5"  # cmake version

    def generate(self):
        tc = CMakeToolchain(self)
        tc.user_presets_path = False

        tc.variables["CONAN_USER_MARCH_FLAGS"] = self._get_march_flags()
        tc.generate()

        cmake = CMakeDeps(self)
        cmake.generate()

    def requirements(self):
        self.requires("gtest/1.16.0")
        self.requires("concurrentqueue/1.0.4")
        self.requires("pybind11/2.13.6")
        self.requires("spdlog/1.14.0")
        self.requires("fmt/10.2.1")  # depends on spdlog
        self.requires("libcoro/0.14.1")

    def configure(self):
        # libcore setting
        self.options["libcoro"].feature_networking = False
        self.options["libcoro"].feature_tls = False
        self.options["libcoro"].build_examples = False
        self.options["libcoro"].build_tests = False

    def package(self):
        self.copy("*.h", dst="include", src="include")
        self.copy("*.hpp", dst="include", src="include")

    def _get_march_flags(self):
        os = self.settings.os
        arch = self.settings.arch
        compiler = self.settings.compiler

        if compiler == "msvc":
            if arch == "x86_64":
                return "/arch:AVX2"
            elif arch == "armv8":
                return "/ARM64"
        elif os in ["Linux", "Macos"]:
            if arch == "x86_64":
                return "-march=x86-64"
            elif arch == "armv8":
                return "-march=armv8-a"
        return ""
