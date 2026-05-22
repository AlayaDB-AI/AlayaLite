# Laser on-disk Quantized Graph (alayalite.laser)

Laser is AlayaLite's on-disk Quantized Graph (QG) index for large-scale
ANN search. It is a port of the `symqglib` reference implementation that
backs the paper *"Efficient Index Layout and Search Strategy for Large-scale
High-dimensional Vector Similarity Search"*. The in-tree code lives under
`include/index/graph/laser/` and `python/src/alayalite/laser/`.

Treat Laser as a vertical port: the SIMD-friendly disk layout, FastScan
4-bit quantization, PCA-reduced coordinates, and asynchronous disk search are
coupled. Changing one part can invalidate the paper-alignment assumptions.

## Platform

Laser is supported on:

- Linux x86_64, using the `libaio` backend by default.
- macOS (arm64 and x86_64), using the portable thread-pool backend by default.
- Windows x64 (MSVC 2022), using the IOCP (I/O Completion Ports) backend by
  default.

Linux ARM is not supported yet — it is explicitly gated by CMake and tracked
in a separate change. Build support is gated by the CMake option
`ALAYA_ENABLE_LASER` (ON by default on Linux x86_64, macOS, and Windows x64;
OFF elsewhere).

```bash
# Debian / Ubuntu
sudo apt-get install libaio-dev

# Fedora / RHEL
sudo dnf install libaio-devel

# macOS (Homebrew)
brew install libomp

# Windows: MSVC 2022 ships the OpenMP 2.0 runtime (vcomp140.dll) as part of
# the C++ workload. No extra package install is required.
```

If `libaio` is missing while Laser is enabled on Linux x86_64, CMake fails
with a message that names the dependency and suggests either
`-DALAYA_ENABLE_LASER=OFF` or `-DALAYA_LASER_USE_THREADPOOL=ON`.

## Input

The public build path consumes base vectors in DiskANN `.fbin` format:
`<int32 N><int32 dim>` followed by `N * dim` row-major `float32` values.
`alayalite.laser.Index.fit(...)` writes all intermediate and final
artifacts under the selected output prefix:

- `<prefix>_pca_base.fbin`
- `<prefix>_pca.bin` when PCA is needed
- `<prefix>_medoids` and `<prefix>_medoids_indices`
- `<prefix>_vamana_graph.index`
- `<prefix>_R<R>_MD<main_dim>.index`

## Build

```bash
cd AlayaLite
uv sync --group laser
```

The `laser` dependency group installs the Python-side runtime needed for
building/searching Laser indexes (`scikit-learn`, `faiss-cpu`, `psutil`,
and supporting packages).

On macOS, install OpenMP before configuring:

```bash
brew install libomp
```

CMake searches the standard Homebrew prefixes (`/opt/homebrew/opt/libomp` on
Apple Silicon and `/usr/local/opt/libomp` on Intel Macs). If OpenMP is not
found while `ALAYA_ENABLE_LASER=ON`, configuration fails with a message naming
`brew install libomp`.

Backend selection is automatic for normal builds:

```bash
# Linux x86_64 default
cmake -B build/Release -DALAYA_ENABLE_LASER=ON
# prints: LASER I/O backend: libaio (Linux x86_64)

# macOS default
cmake -B build/Release -DALAYA_ENABLE_LASER=ON
# prints: LASER I/O backend: thread pool (macOS)

# Windows x64 default
cmake -B build/Release -G "Visual Studio 17 2022" -A x64 -DALAYA_ENABLE_LASER=ON
# prints: LASER I/O backend: IOCP (Windows x64)

# Linux fallback without libaio
cmake -B build/Release -DALAYA_ENABLE_LASER=ON -DALAYA_LASER_USE_THREADPOOL=ON
# prints: LASER I/O backend: thread pool (portable fallback)
```

The thread-pool backend uses buffered `pread` and worker threads. Override the
worker count for local experiments with `ALAYA_LASER_IO_THREADS`; production
Linux x86_64 builds should keep the default libaio backend unless the portable
fallback is intentionally being tested.

The Windows IOCP backend uses `CreateFileW(FILE_FLAG_NO_BUFFERING |
FILE_FLAG_OVERLAPPED)` + a single I/O completion port. A dedicated
dispatcher thread drains the port via `GetQueuedCompletionStatusEx` and
routes completions to the originating consumer thread's queue. The 4096-byte
alignment contract on `AlignedRead` is enforced by `FILE_FLAG_NO_BUFFERING`'s
sector-size requirement. Builds require MSVC 2022; OpenMP is provided by the
default MSVC `/openmp` flag, which CMake's `find_package(OpenMP)` picks up
automatically. LASER's `#pragma omp parallel for` / `critical` / `schedule`
usage stays inside the OpenMP 2.0 surface, so the legacy MSVC OpenMP runtime
(`vcomp140.dll`, shipped with VC Redist) is sufficient.

## SIMD Dispatch

LASER's handwritten SIMD kernels select their ISA at runtime. A single x86_64
wheel carries AVX-512F+BW and AVX2+FMA implementations for FastScan,
approximate-distance conversion, rotation, scalar range scans, and single-vector
L2 norm. On CPUs with both AVX-512F and AVX-512BW, LASER selects the AVX-512
path; otherwise it uses AVX2+FMA. Non-x86 builds use the generic scalar LASER
path; this is intended for macOS development portability, not throughput parity.

The baseline `-mavx2 -mfma` flags are still used for Eigen and other
non-handwritten code. Function-level target attributes do not change Eigen's
template-selected SIMD path, so PCA matrix-vector work remains tied to the wheel
baseline even on AVX-512 hosts. The expected impact is limited to that Eigen
portion of the search path.

On MSVC the LASER consumer flags become `/arch:AVX2 /DEIGEN_DONT_PARALLELIZE`
in place of `-mavx2 -mfma -ftree-vectorize`. MSVC auto-vectorizes under `/O2`
by default, so the GCC `-ftree-vectorize` flag has no MSVC analogue and is
omitted from the Windows compiler line.

## Performance Notes

- **Huge pages**: The Linux backend hints the kernel with
  `madvise(MADV_HUGEPAGE)` on the LASER read scratch buffers. macOS and
  Windows are no-ops by design. On Windows, `VirtualAlloc(MEM_LARGE_PAGES)`
  requires the calling process to hold `SeLockMemoryPrivilege` (an admin-
  granted right not present by default on wheel-consumer machines), and
  kernel large-page allocation has noisy availability after system uptime.
  Power users who want to enable large pages locally can grant the
  privilege via Group Policy (`secpol.msc`) — but this is not the
  documented wheel path. Expect a modest hit on search throughput when
  the working set exceeds the small-page TLB (single-digit percent on
  100M-vector indexes; not measurable on 100K-vector smoke runs).

- **Linux vs Windows perf parity**: The cross-platform perf workflow
  (`laser-cross-platform-perf`) runs the same synthetic dataset against
  both Linux libaio and Windows IOCP. Significant (>20% same-recall QPS)
  deltas should be investigated before tagging a release — but search
  recall itself is expected to match within ±1pp (D7 contract:
  search-side parity is statistical, not byte-equal, because thread-pool
  / IOCP completion ordering is non-deterministic across backends).

- **Sector size**: All three backends now use a 4096-byte alignment
  contract (raised from the previous 512 on Linux). 4K matches the LASER
  on-disk page layout and is superset-acceptable for both 512n and 4Kn
  drives. Consumer code does not need to special-case any backend.

## Python API

Use the unified wrapper for normal code:

```python
from alayalite.laser import BuildParams, Index

idx = Index.fit(
    "/path/to/base.fbin",
    output_dir="/path/to/build/laser",
    name="dsqg_gist",
    build_params=BuildParams(
        metric="l2",
        main_dim=256,
        R=64,
        L=200,
        alpha=1.2,
        ef_indexing=200,
        ep_num=300,
    ),
    seed=42,
    num_threads=48,
    dram_budget_gb=32.0,
)

idx.set_params(ef_search=200, num_threads=1, beam_width=16)
ids = idx.batch_search(queries, 10)
```

To reopen an existing build:

```python
idx = Index.from_prefix("/path/to/build/laser/dsqg_gist", dram_budget_gb=1.0)
```

The raw pybind class is still exported as `alayalite.laser.RawIndex` for
research code that intentionally bypasses the wrapper.

## CLI

`examples/laser/main.py` is a compact TOML wrapper around `Index.fit`
plus an EF sweep:

```bash
# Build, then run the configured EF sweep.
uv run examples/laser/main.py -c examples/laser/configs/gist.toml all

# Build only.
uv run examples/laser/main.py -c examples/laser/configs/gist.toml build

# Search only against an existing build.
uv run examples/laser/main.py -c examples/laser/configs/gist.toml search \
    --threads 1 --efs 100 200 300
```

The CLI exposes only `build`, `search`, and `all`. The older
`vamana/pca/medoid/index` step names are retired.

### TOML Shape

```toml
seed = 42

[dataset]
name = "gist"
metric = "l2"
degree = 64
main_dimension = 256

[paths]
base = "/path/to/base.fbin"
query = "/path/to/query.fbin"
gt = "/path/to/gt.ibin"
output = "/path/to/build/laser"

[build]
build_threads = 48
ef_indexing = 200

[build_vamana]
L = 200
alpha = 1.2
dram_budget_gb = 32.0

[search]
topk = 10
threads = 1
beam_width = 16
dram_budget = 1.0
ep_num = 300
warmup = 10
runs = 30
efs = [100, 200, 300]
```

`[dataset].degree` is the single source of truth for Vamana `R` and the
Laser degree bound. `examples/laser/main.py` rejects stale fields that
belonged to the old step-by-step reproduction pipeline, including
`[paths].vamana`, per-step seed fields, `dump_rotator`, and
`[build_vamana].seed/num_threads`.

## Reproducibility

`seed` is the master seed used by the unified wrapper for randomized
build sub-steps. Set `build_threads = 1` and run under single-threaded
BLAS/OpenMP settings when you need the most stable local rebuilds.

The old cross-repo Tier A byte-equality harness no longer drives
`examples/laser/main.py` through individual build stages. It is retained
as an existing-artifact comparator at
`scripts/laser_alignment/tier_a_byte_equal.py`.

## Vamana Low-Level API

The standalone Vamana builder remains available for non-Laser workflows:

```python
from alayalite import vamana

vamana.build_index(
    data_path="/path/to/base.fbin",
    output_path="/path/to/graph.index",
    R=64,
    L=200,
    alpha=1.2,
    seed=42,
    num_threads=48,
    dram_budget_gb=32.0,
)
```

Laser's public wrapper builds its own Vamana artifact, so this low-level
entry point is not part of the `examples/laser` CLI contract.

## Known Issues

### Fixed: Low-Dimensional Page Layout

`fix-laser-low-dim-page-layout` fixes the upstream LASER page-layout
mismatch in `include/index/graph/laser/qg/qg_builder.hpp`: the build path
now packs up to `node_per_page_` consecutive node payloads into each
`page_size_` page, matching the `qg.hpp` read formulas
`get_page_offset(id)` and `offset_to_node(id)`.

For `node_per_page_ == 1` configurations, the layout reduces to the
original one-node-per-page output. For `node_per_page_ > 1`
configurations, including SIFT-1M-style `main_dim=64` builds, construction
is no longer refused for the historical write/read mismatch. This fix is
tracked by PR #88.

### Relationship To AlayaLite QG

AlayaLite also has a separate QG builder under
`include/index/graph/qg/qg_builder.hpp`. That path uses a different
quantization flow and is not the same component as the Laser port under
`include/index/graph/laser/`.
