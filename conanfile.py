from conan import ConanFile 
from conan.tools.cmake import cmake_layout


class AlayaLiteConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"
    package_type = "header-library"
    exports_sources = "include/*"
    platform_tool_requires = "cmake/3.23.5" # cmake version

    def requirements(self):
        self.requires("gtest/1.16.0")
        self.requires("concurrentqueue/1.0.4")
        self.requires("pybind11/2.13.6")
        self.requires("spdlog/1.14.0")
        self.requires("fmt/10.2.1") # depends on spdlog
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
