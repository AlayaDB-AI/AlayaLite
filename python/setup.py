import os
import pathlib
import platform
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ALAYA_ROOT = os.environ.get("ALAYA_ROOT")
if ALAYA_ROOT is None:
    ALAYA_ROOT = pathlib.Path(__file__).parent.parent.resolve()
else:
    ALAYA_ROOT = pathlib.Path(ALAYA_ROOT).resolve()
ALAYA_CMAKE_DIR = pathlib.Path(ALAYA_ROOT, "build")


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(pathlib.Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def finalize_options(self):
        # not use
        if sys.platform.lower() == "linux":
            self.plat_name = f"manylinux2014_{platform.machine().lower()}"
        elif sys.platform.lower() == "darwin":
            if platform.machine().lower() == "arm64":
                self.plat_name = f"macosx_11_0_{platform.machine().lower()}"
            else:
                self.plat_name = f"macosx_10_9_{platform.machine().lower()}"
        else:
            ...  # Windows or other platforms
        return super().finalize_options()

    def _build_conan(self, extdir, build_lib, build_temp, os_output_dir, env):
        extdir = os.path.abspath(extdir)
        build_lib = os.path.abspath(build_lib)
        build_temp = os.path.abspath(build_temp)

        # generate conan profile
        subprocess.check_call(
            ["conan", "profile", "detect", "-e"],
            cwd=build_temp,
            env=env,
        )

        if sys.platform.lower() == "linux":
            subprocess.check_call(
                [
                    "conan",
                    "install",
                    extdir,
                    "--build=missing",
                    "--output-folder=generate",
                    "--settings=build_type=Release",
                    "--settings=compiler.cppstd=20",
                ],
                cwd=build_temp,
                env=env,
            )
        else:
            ...
        subprocess.check_call(
            [
                "cmake",
                extdir,
                "-DENABLE_UNIT_TESTS=OFF",  # Disable c++ unit tests for the extension build
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os_output_dir}",
                "-DCMAKE_TOOLCHAIN_FILE=generate/conan_toolchain.cmake",
                "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            cwd=build_temp,
            env=env,
        )

        subprocess.check_call(
            ["cmake", "--build", ".", "--", "-j"],
            cwd=build_temp,
            env=env,
        )

    def run(self):
        for ext in self.extensions:
            extdir = os.path.abspath(ALAYA_ROOT)
            build_temp = os.path.abspath(
                ALAYA_CMAKE_DIR
            )  # if use self.build_temp, it will delete the build directory when build_ext is done
            os_output_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            env = os.environ
            os.environ["CONAN_CMAKE_USER_PRESETS"] = "False"
            os.makedirs(build_temp, exist_ok=True)
            self._build_conan(extdir, self.build_lib, build_temp, os_output_dir, env)


setup(
    ext_modules=[CMakeExtension("alayalite._alayalitepy")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    package_dir={"": "src"},
)
