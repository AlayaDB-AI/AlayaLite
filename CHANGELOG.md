# Changelog

All notable changes to AlayaLite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `disk-segment-searcher-dispatch`: refactored `DiskCollection` to dispatch
  segment construction through a new `disk-segment-factory` layer
  (`include/index/disk/segment_factory.hpp`). Five sites that previously
  hard-coded `DiskFlatBuilder` / `DiskFlatSegmentSearcher` now route through
  the factory's `engine_supported_v1` / `create_segment_from_pending` /
  `load_segment_from_manifest` entry points. Cross-segment label
  uniqueness is now engine-agnostic (mmap'd `manifest.ids_file` reads
  instead of `dynamic_cast<DiskFlatSegmentSearcher *>`). Flat behaviour,
  byte format, and Python API are unchanged. Laser remains rejected at the
  v1 capability gate with the same dual-substring error contract (engine
  name + "not implemented in v1").

### Added
- DiskCollection now supports the LASER engine end-to-end at the C++ level for
  load+search+import (L2 only; Linux + libaio +
  `ALAYA_ENABLE_LASER=ON` only): `index_type=disk_laser` reachable through
  `DiskCollection`, segment importer + searcher under `include/index/disk/`,
  and a SegmentFactory registration. Native LASER files and sidecars are
  co-located inside `seg_<id>/` under their native filenames and recorded in
  `manifest.x_extras`. v1 has no in-C++ build pipeline; segments enter via
  `DiskCollection::import_laser_segment` from precomputed artifacts produced by
  the upstream Python module + `QGBuilder::build`. Python `disk_laser` exposure
  remains deferred to a follow-up.
- DiskCollection now supports the Vamana engine end-to-end (L2 only):
  `index_type="disk_vamana"` at the C++ level, segment builder + searcher
  under `include/index/disk/`, and a SegmentFactory registration. Python
  `disk_vamana` exposure remains deferred to a follow-up.
- Disk-resident segmented collection (`disk-collection` + `disk-flat-builder`
  + `disk-flat-searcher` + `disk-types` + `mmap-file` + `segment-manifest`
  capabilities). New `alayalite.DiskCollection` Python surface (constructor
  + static `open(path)` factory + `add` / `flush` / `search` / `dim` /
  `size`), wrapping `alaya::disk::DiskCollection` in C++. v1 supports the
  Flat (brute-force) segment type with L2, IP, and COS metrics; Vamana and
  Laser segment types are reserved enum values rejected at the v1
  capability gate. On-disk layout: `<collection>/segments/seg_NNNNNNNN/`
  containing `manifest.txt`, `ids.u64.bin`, and `vectors.f32.bin`, plus a
  top-level `collection_manifest.txt`. POSIX-only (Linux + macOS); Windows
  builds compile but throw at runtime. See
  `openspec/changes/archive/2026-04-30-add-disk-collection-flat/` and
  `examples/disk_collection_basic.py`.
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
- `scripts/laser_alignment/gen_synth_100k_512d.py` — synthetic dataset
  generator used as a secondary alignment judge for the Laser port.
- Sharded Vamana partition-merge alignment with patched upstream
  DiskANN (`diskann-sharded-alignment-gate` capability). Tier A
  asserts byte-equality on `_medoids.bin` (the partition-stage
  invariant) and structural parity on the other artifact classes
  between AlayaLite's `build_vamana_index` CLI and the patched
  DiskANN `build_merged_vamana_standalone` CLI at matched seeds on
  `synth_100k_512d`. Test driver: `tests/vamana/test_sharded_byte_equality.py`.
  Exposed `BuildVamanaParams::sampling_rate` (sentinel auto =
  `min(1.0, 256000/N)`) so the partition growth loop is numerically
  stable on small datasets. Tier B retained as nightly statistical
  envelope; harness now accepts `--expected_num_parts_envelope <lo> <hi>`.
  See `openspec/changes/archive/2026-04-25-align-diskann-sharded-with-upstream/`.

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
