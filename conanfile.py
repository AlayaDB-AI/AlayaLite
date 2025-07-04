from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps


class AlayaLiteConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    package_type = "header-library"
    exports_sources = "include/*"
    platform_tool_requires = "cmake/3.23.5"  # cmake version

    def generate(self):
        tc = CMakeToolchain(self)
        tc.user_presets_path = False

        CONAN_USER_MARCH_FLAGS = ""
        if self.settings.os == "Linux" or self.settings.os == "Macos":
            if self.settings.arch == "x86_64":
                CONAN_USER_MARCH_FLAGS = "-march=x86-64"
            elif self.settings.arch == "armv8" or self.settings.arch == "aarch64":
                CONAN_USER_MARCH_FLAGS = "-march=armv8-a"
        elif self.settings.os == "Windows":
            # TODO: add flag for msvc
            if self.settings.compiler == "msvc": ...
        else:
            self.output.info(f'Unknown OS: {self.settings.os}, skipping setting CONAN_USER_MARCH_FLAGS')

        tc.variables["CONAN_USER_MARCH_FLAGS"] = CONAN_USER_MARCH_FLAGS

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
