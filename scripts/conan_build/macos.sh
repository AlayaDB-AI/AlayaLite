#!/usr/bin/env bash
set -euo pipefail

# 1. Configure compiler (for macOS environment, use gcc/g++ for Linux, ignore for Windows)
export CC=clang
export CXX=clang++

# 2. Detect and refresh conan profile to ensure correct compiler settings
conan profile detect --force

# 3. Get clang version (assuming clang, automatically extract major version)
CLANG_VERSION=$(clang --version | grep -o 'Apple clang version [0-9]*' | grep -o '[0-9]*' | head -n1)
if [ -z "$CLANG_VERSION" ]; then
  echo "Failed to detect clang version, defaulting to 15"
  CLANG_VERSION=15
fi

# 4. Define output directory (CMake recommends using build/conan_generate)
CONAN_OUT_DIR=build/conan_generate
mkdir -p "$CONAN_OUT_DIR"

# 5. Run conan install
conan install . \
  --output-folder="$CONAN_OUT_DIR" \
  --build=missing \
  --settings=build_type=Release \
  --settings=compiler=apple-clang \
  --settings=compiler.version="$CLANG_VERSION" \
  --settings=compiler.libcxx=libc++ \
  --settings=compiler.cppstd=20

echo "Conan install finished. Toolchain file at $CONAN_OUT_DIR/conan_toolchain.cmake"