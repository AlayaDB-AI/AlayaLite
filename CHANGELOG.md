# Changelog

All notable changes to AlayaLite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LASER on-disk index is now buildable and runnable on Windows x64 via the
  new IOCP (I/O Completion Ports) I/O backend
  (`include/index/graph/laser/utils/iocp_file_reader.hpp`). The release
  wheel matrix now ships native LASER on all 5 platform lanes
  (`manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`, `macos x86_64`,
  `macos arm64`, `windows-2022 AMD64`); Linux ARM remains the only lane
  with LASER off-by-default (tracked in a separate change). The Windows
  lane uses MSVC 2022 with the toolchain-default `/openmp` runtime
  (`vcomp140.dll`); LASER stays inside OpenMP 2.0 surface, no
  vcpkg/Conan OpenMP package is required. `_alayalitepy.pyd` runtime
  DLLs are bundled into the wheel by `delvewheel`.
- LASER on-disk index is now buildable and runnable on macOS (M-series and
  Intel) using a portable thread-pool I/O backend. Linux x86_64 continues to
  use the libaio backend with no behavior change.

### Changed
- Disk/storage/space layer no longer pulls POSIX headers
  (`<unistd.h>`, `<sys/mman.h>`, `<sys/file.h>`, ...) unconditionally —
  the LAYER 2 portability sweep routed every previously-POSIX file
  operation through `include/utils/platform_fs.hpp` helpers
  (`write_all_fsync`, `read_regular_file_bounded`, `read_file_prefix`,
  `atomic_replace_no_overwrite`, `sync_file_or_throw`,
  `sync_directory_or_throw`, `truncate_file`, `get_pid`,
  `is_readable_regular_file`). The collection lock acquisition in
  `disk_collection.hpp` is now per-platform: POSIX keeps the existing
  `flock` + inode-swap defense, Windows uses
  `LockFileEx(LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY)` plus
  shared-mode flags that exclude DELETE (the kernel-level analogue of
  the POSIX KL3 defense).
- `MMapFile` (`include/storage/mmap_file.hpp`) replaced its
  `throw "unsupported"` Windows stub with a real
  `CreateFileMappingW + MapViewOfFile` implementation.
- LASER sector-size constant raised from 512 to 4096 across all backends
  (matches the LASER on-disk page layout; 4Kn-drive compatible without
  reducing 512n correctness).
- LASER memory.hpp prefetch hint detection no longer relies on the
  `__SSE2__` macro (which MSVC does not define); uses
  `ALAYA_ARCH_X86`-gated `_mm_prefetch` with a `__builtin_prefetch`
  fallback that is itself gated on GCC/Clang availability.

### Fixed
- LASER SIMD rotation and scalar range kernels now use unaligned load/store
  instructions in the runtime-dispatched paths, avoiding potential aligned
  AVX-512 stores into Eigen buffers that are only 32-byte aligned under the
  wheel's AVX2 baseline.
- Disk LASER index now supports `node_per_page_ > 1` (low-dim datasets like
  SIFT-1M); previously refused at construction. Fixes the upstream
  `qg_builder.hpp` write/read page-layout mismatch.
- Disk LASER build no longer zero-clobbers per-node neighbor IDs. The
  `update_qg_out_of_memory` path used to write neighbor IDs into the per-node
  payload **before** the `read()` that pulled the centroid vector into the
  same `page_buf`; for any `node_len_` that fit inside a single sector
  (SIFT-1M `main_dim=64`), the read overwrote the neighbor region with the
  zero-padded tail of the temporary vector file, leaving every node with
  `neighbors[*] = 0` and collapsing recall to random level. Lifts SIFT-1M
  `recall@10` from 0.0009 to 0.9984 at `ef_search=800`. `main_dim=256`
  configurations (gist1m, bigcode, synth_20k_768d) were never affected, since
  `neighbor_offset_ > full_page_size` there.

### Changed
- LASER handwritten SIMD kernels now use runtime dispatch: wheels keep the
  AVX2+FMA x86 baseline while selecting AVX-512F+BW kernels automatically on
  capable hosts.
- `pybind-dispatch-codegen`: replaced the hand-written
  `python/include/dispatch.hpp` macro dispatch chain with a codegen-driven
  pipeline. Single source of truth lives in `tools/codegen/dispatch.yaml`,
  expanded by `tools/codegen/gen.py` into
  `python/include/dispatch_generated.hpp` (out-of-line `IndexFactory::create`
  body) and `python/tests/client/_dispatch_matrix_params.py` (parametrized
  test inputs). `IndexFactory::create` lives in its own translation unit
  (`python/src/index_factory.cpp`) so the 33 template specializations land in
  one `.o`; `_alayalitepy.so` drops from ~43M to ~26M (-40%). New
  `make codegen` target regenerates both files. The two generated files are
  excluded from pre-commit format/lint hooks (clang-format / ruff / pylint /
  cpplint) so `gen.py` has no build-time dependency on clang-format.
  **Breaking**: the supported vector dtype matrix is narrowed to
  `{np.float32, np.int8, np.uint8}` — `np.float64`, `np.int32`, `np.uint32`
  are no longer accepted at the engine boundary (`alayalite.common.valid_dtype`
  raises `ValueError`). The previous wider matrix was declared in macros but
  rarely exercised end-to-end; downstream callers must cast to `np.float32`
  (the recommended default) before `fit` / `insert` / `search`.
- `disk-collection-single-writer-lock`: `DiskCollection` now holds a
  process-level single-writer lock at `<collection>/.lock`; concurrent opens
  of the same collection raise with a stable dual-substring error containing
  the lock path and `collection is already open by another process`. NFS /
  Windows remain unsupported; no reader-writer lock and no WAL are introduced.
- `disk-collection-ctor-create-atomicity`: closes the three residual
  concurrency windows acknowledged by `disk-collection-single-writer-lock`.
  The `DiskCollection` constructor now uses an `O_CREAT|O_EXCL` acquire
  helper (`acquire_collection_lock_for_create`) that serializes concurrent
  ctors at the kernel layer and rolls back its just-created `path/`
  directory on acquire failure. `DiskCollection::open(path)` rejects a
  half-published target (`path/` exists but neither `.lock` nor
  `collection_manifest.txt` are present) with the dual-substring error
  containing the `.lock` path and `target path is a collection-in-progress,
  not yet published`. Both acquire entry points run a post-flock
  `(st_dev, st_ino)` revalidation against `fstat(fd)` so an in-flight
  `unlink(.lock); touch .lock` swap throws `lock file inode mismatch
  after acquire`. EEXIST on the ctor entry surfaces `target path already
  exists or is being created concurrently`. Out-of-band `.lock`
  manipulation that completes before a new acquire begins remains an
  explicit non-goal (no stat-based mechanism can observe it).
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
- Add unified DiskCollection benchmark harness
  (`python -m alayalite.bench.disk_collection`) covering disk_flat /
  disk_vamana / disk_laser engines with runtime-gated laser support and
  provenance-bearing JSON+MD output.
- expose `disk_laser` via `alayalite.DiskCollection` Python binding (import +
  search; v1 does not support add/flush/batch_search). Lifts the binding-side
  hard veto so `index_type="disk_laser"` is gated by the C++
  `engine_supported_v1(Laser)` predicate (Linux + `ALAYA_ENABLE_LASER=ON` +
  libaio). Adds `DiskCollection.import_laser_segment(src_dir, labels, *,
  copy=True)` driving the C++ importer with binding-side validation
  (src_dir directory check, labels dtype/contig/shape, GIL release). Threads
  `beam_width` (keyword-only, default 4) through `search()`, applied
  uniformly across engines. Adds an engine-uniform NaN / Inf check on
  `query` that uses the bit-pattern `is_finite_f32` helper (per
  `project_ofast_finiteness_check`). Rejects `add()` / `flush()` on
  `disk_laser` collections at the binding boundary with the dual-substring
  contract pointing at `import_laser_segment`. Non-goals: no in-C++ build
  pipeline, no add/flush/batch_search, no PCA/medoid C++ port, no LASER
  format change, no metadata filter, no WAL/compaction/delete/upsert. v1's
  `copy=False` raises `NotImplementedError` (the C++ entry point does not
  accept a per-call params override; the keyword is preserved on the API
  surface for a future change). On unsupported configurations
  (Linux+OFF / macOS / Windows / SIMD-unsupported Linux) the wheel keeps
  building and `disk_laser` raises `ValueError` with the dual-substring
  message; positive-path Python tests are gated on a runtime probe so the
  test matrix stays green across the wheel matrix.
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
