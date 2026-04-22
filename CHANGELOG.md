# Changelog

All notable changes to AlayaLite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Disk-based vector index (DiskANN) for billion-scale datasets
- Real-time update support for dynamic vector insertion and deletion
- Scalar-vector fusion search for hybrid queries
- Laser on-disk Quantized Graph index (`laser-disk-index` capability).
  New `alayalite.laser.Index` Python surface, CLI pipeline at
  `examples/laser/`, and the `alaya::laser::QuantizedGraph` C++ class
  under `include/index/graph/laser/`. Consumes DiskANN-format Vamana
  `.index` + `.fbin` inputs; produces a FastScan + RabitQ quantized
  on-disk layout served via `libaio` beam search. See `docs/LASER.md`.
- Build-time `ALAYA_ENABLE_LASER` CMake option (default ON on Linux).
  Adds a new system build dependency on `libaio-dev` for Linux builds
  when the option is ON.
- `scripts/gen_synth_100k_512d.py` — synthetic dataset generator used
  as a secondary alignment judge for the Laser port.

## [0.1.1-alpha1] - 2026-01-28

### Added
- Full cibuildwheel support for multi-platform builds:
  - Linux: x86_64, aarch64
  - macOS: x86_64 (Intel), arm64 (Apple Silicon)
  - Windows: x86_64
- Unified Conan dependency management script (`conan_install.py`)
- pylint integration for Python code quality

### Changed
- Simplified CMake build system
- Updated type annotations for Python 3.8 compatibility
- Configured static linking for all Conan dependencies

## [0.1.0-alpha3] - 2026-01-15

### Added
- RaBitQ (Random Bit Quantization) implementation
- Standalone app with reset functionality
- C++ code coverage support (Codecov)
- Custom index parameters in Collection class
- SetMetricRequest endpoint for collection metric configuration

### Changed
- Refactored ANN-benchmark adaptation
- Modularized CMake configuration
- Improved RAG example error handling

### Fixed
- RAG example embeddings check failure
- Multiple bug fixes and code refactoring
- AlayaLite wheel URL in Dockerfile

## [0.1.0-alpha2] - 2026-01-01

### Added
- Python code coverage (Codecov)
- Typo's pre-commit check
- Initial standalone app implementation

### Changed
- Updated documentation with absolute URLs for online access
- Added icons to documentation

## [0.1.0-alpha1] - 2025-12-15

### Added
- Initial release of AlayaLite
- HNSW (Hierarchical Navigable Small World) index
- NSG (Navigating Spreading-out Graph) index
- Fusion Graph index
- SQ8 (8-bit scalar quantization) support
- L2, Inner Product, and Cosine similarity metrics
- Python SDK with Client, Index, and Collection APIs
- RAG components (embedders, chunkers)
- Basic CI/CD pipeline

[Unreleased]: https://github.com/AlayaDB-AI/AlayaLite/compare/v0.1.1a1...HEAD
[0.1.1-alpha1]: https://github.com/AlayaDB-AI/AlayaLite/compare/v0.1.0a3...v0.1.1a1
[0.1.0-alpha3]: https://github.com/AlayaDB-AI/AlayaLite/compare/v0.1.0a2...v0.1.0a3
[0.1.0-alpha2]: https://github.com/AlayaDB-AI/AlayaLite/compare/v0.1.0a1...v0.1.0a2
[0.1.0-alpha1]: https://github.com/AlayaDB-AI/AlayaLite/releases/tag/v0.1.0a1
