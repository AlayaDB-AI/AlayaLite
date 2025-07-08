#!/usr/bin/env bash
set -euo pipefail

# 1. Configure compiler (typically gcc/g++ for Linux)
export CC=gcc
export CXX=g++

# 2. Detect and refresh conan profile to ensure correct compiler settings
conan profile detect --force

# 3. Get gcc major version
GCC_VERSION=$(gcc -dumpfullversion | cut -d. -f1)
if [ -z "$GCC_VERSION" ]; then
  echo "Failed to detect gcc version, defaulting to 11"
  GCC_VERSION=11
fi

# 4. Define output directory
CONAN_OUT_DIR=build/conan_generate
mkdir -p "$CONAN_OUT_DIR"

# 5. Run conan install
conan install . \
  --output-folder="$CONAN_OUT_DIR" \
  --build=missing \
  --settings=build_type=Release \
  --settings=compiler=gcc \
  --settings=compiler.version="$GCC_VERSION" \
  --settings=compiler.libcxx=libstdc++11 \
  --settings=compiler.cppstd=20

echo "Conan install finished. Toolchain file at $CONAN_OUT_DIR/conan_toolchain.cmake"