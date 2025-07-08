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

        march_flags = ""
        os = str(self.settings.os)
        arch = str(self.settings.arch)
        compiler = str(self.settings.compiler)
        if compiler == "msvc":
            if arch == "x86_64":
                march_flags = "/arch:AVX2"
        elif os in ["Linux", "Macos"]:
            if arch == "x86_64":
                march_flags = "-march=x86-64-v2"
            elif arch in ["armv8", "aarch64", "arm64"]:
                march_flags = "-march=armv8-a"

        if march_flags:
            self.output.info(f"Applying architecture flag for {os}/{arch}: {march_flags}")
            tc.variables["CONAN_USER_MARCH_FLAGS"] = march_flags
        else:
            self.output.warning(f"No specific march flag set for this configuration ({os}/{arch}).")

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
