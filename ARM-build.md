# ARM Platform Build Guide

## System Requirements
- **CMake** >= 3.1, <= 3.31
- **Ninja** build system (recommended)
- **ARMv7+/ARM64** compatible toolchain
- **Python 3.10+** (for Python bindings)
- **GCC** gcc10+ (gcc13 recommended)

## Build Methods

### 1. Python Bindings (Recommended)
```bash
./build_support/pyinstall.sh
```

### 2. CMake Build (Makefile)

```bash
# Configure and build
mkdir -p build && cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DNO_COROUTINE=True \
  ..
make -j$(nproc)
```
### 3. CMake Build (Ninja - Faster)
```bash
mkdir -p build && cd build
cmake \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNO_COROUTINE=ON \
  ..
ninja
```
