# Laser on-disk Quantized Graph (alayalite.laser)

Laser is AlayaLite's on-disk Quantized Graph (QG) index for billion-scale
ANN search. It is a port of the `symqglib` reference implementation that
backs the paper *"Efficient Index Layout and Search Strategy for Large-scale
High-dimensional Vector Similarity Search"*; the original repo lives at
`/md1/huangliang/alaya-dev/Laser` and the in-tree port lives under
`include/index/graph/laser/` (C++) + `python/src/alayalite/laser/` (Python).

The design co-designs three pillars: (1) a SIMD-friendly interleaved
on-disk layout, (2) FastScan 4-bit quantization on PCA-reduced coordinates,
(3) `libaio`-driven async I/O with beam expansion and long-tail mitigation.
Breaking any one invalidates the paper's claims — treat the port as a
vertical copy, not a refactor opportunity.

## Platform

Linux only. `libaio` is a thin wrapper over the `io_submit` / `io_getevents`
kernel syscalls; macOS and Windows have no direct equivalent. The build
gate is the CMake option `ALAYA_ENABLE_LASER` (defaults ON on Linux, OFF
elsewhere). Non-Linux builds skip the Laser target silently.

### Build-time system dependency

```bash
# Debian / Ubuntu
sudo apt-get install libaio-dev

# Fedora / RHEL
sudo dnf install libaio-devel
```

If `libaio` is missing the CMake run fails with an explicit message
pointing at these commands or at the `-DALAYA_ENABLE_LASER=OFF` escape
hatch.

## Input contract

The Laser index consumes two inputs:

1. **Base vectors**, DiskANN `.fbin` format:
   `<int32 N><int32 dim>` header, then `N × dim` `float32` row-major.
2. **Vamana graph**, DiskANN `.index` format, produced by any of:
   * the Python binding `alayalite.vamana.build_index(...)` (integrated
     pipeline — preferred for end-to-end reproduction);
   * AlayaLite's `build_vamana_for_laser` CLI (see
     `tools/build_vamana_index/`) — equivalent output, retained as the
     reference trajectory for alignment research against DiskANN's
     `search_memory_index`;
   * DiskANN upstream's `build_memory_index` (external trajectory, used
     by `examples/laser/configs/gist_diskann.toml`).

The Python binding and the CLI share one dispatch library
(`include/index/graph/vamana/build_dispatch.hpp`); given identical
parameters and `num_threads=1`, they produce byte-for-byte identical
outputs — see `tests/vamana/test_cli_vs_python_parity.py` (Gate G1).

## Building from source

```bash
# Conan deps + CMake + pybind are all wired through scikit-build.
cd AlayaLite
uv sync --group laser
```

The build produces a single `_alayalitepy.cpython-*.so` that hosts both
the main AlayaLite APIs and a `laser` submodule. `from alayalite.laser
import Index` resolves through `alayalite._alayalitepy.laser.Index`.

To turn the Laser module off (e.g. on macOS or for a CI job without
`libaio-dev`):

```bash
cmake -DALAYA_ENABLE_LASER=OFF ...
```

The CMake configure step errors out with an explicit hint if
`ALAYA_ENABLE_LASER=ON` is requested on a platform without `libaio`.

## Python API

```python
from alayalite import vamana
from alayalite.laser import Index

# Step 1: build the Vamana graph from the raw .fbin. Writes a DiskANN-
# format .index file that Laser's QGBuilder can consume directly.
vamana.build_index(
    data_path="/path/to/base.fbin",
    output_path="/path/to/graph.index",
    R=64,                        # matches Laser's degree_bound below
    L=200,
    alpha=1.2,
    seed=1234,
    num_threads=0,               # 0 → omp_get_num_procs()
    dram_budget_gb=32.0,         # single-shard budget; larger = in-memory build
)

# Step 2: construct the Laser index shell.
index = Index(
    index_type="QG",
    metric="l2",
    num_elements=N,
    main_dimension=256,          # PCA-reduced dim; residual = dim - 256
    dimension=D,                 # original dim
    degree_bound=64,             # must equal the Vamana R above and be a multiple of 32
)

# Step 3: one-shot build. Writes dsqg_<name>_R{deg}_MD{md}.index next to
# data_file. Runs once before search; pipeline example below.
index.build_index(
    vamana_file="/path/to/graph.index",
    data_file="/path/to/output_prefix/dsqg_<name>",
    EF=200,                      # ef_indexing
    num_iter=3,
    num_thread=48,
)

# Or load a pre-built dsqg .index.
index.load("/path/to/output_prefix/dsqg_<name>", search_DRAM_budget=1.0)

# Configure search.
index.set_params(ef_search=200, num_threads=1, beam_width=16)

# Query.
k = 10
ids = index.batch_search(queries, k)   # (NQ, k) uint32
```

The unified `alayalite.laser.Index.fit(...)` runs the full
PCA / medoid / Vamana / QG pipeline in one call; `examples/laser/main.py`
wraps it behind two CLI steps (`build` and `search`). Direct Python API
above remains for callers who want to integrate Laser into a larger
pipeline.

## CLI

The paper-reproduction pipeline sits at `examples/laser/`:

```bash
# Both steps: build → EF sweep search.
uv run examples/laser/main.py -c examples/laser/configs/gist.toml all

# Or just search (assumes an existing index under the config's output dir).
uv run examples/laser/main.py -c examples/laser/configs/gist.toml search \\
    --threads 1 --efs 100 200 300

# Build only (no search).
uv run examples/laser/main.py -c examples/laser/configs/gist.toml build
```

Bundled configs: `gist.toml` (gist1m + AlayaV Vamana), `gist_diskann.toml`
(gist1m + DiskANN-built Vamana), `synth_100k_512d.toml` (synthetic 100K
× 512d dataset generated by `scripts/laser_alignment/gen_synth_100k_512d.py`).

`step_vamana` writes its output to `[paths].vamana` when that field is
set in the TOML, falling back to the derived path
`{[paths].output}/data/{[dataset].name}/vamana/graph.index` otherwise.
Existing configs that pin a pre-built `.index` continue to work —
`step_vamana` validates the file's DiskANN header and skips the build
when `max_observed_degree == [dataset].degree` and `frozen_pts == 0`.

## Vamana graph building

Two entry points produce the DiskANN-format `.index` file that Laser
consumes. They share one dispatch library and one set of defaults, so
their outputs are byte-equal given identical parameters and
`num_threads=1`.

### Python binding: `alayalite.vamana.build_index`

```python
from alayalite import vamana
vamana.build_index(
    data_path="/path/to/base.fbin",
    output_path="/path/to/graph.index",
    R=64, L=200, alpha=1.2, seed=1234,
    num_threads=0, dram_budget_gb=32.0,
)
```

`R` is required and has no default; all other parameters default from
`alaya::vamana::kDefaultVamanaBuildParams` in
`include/index/graph/vamana/build_dispatch.hpp`. Errors surface as:

* `ValueError` for malformed `.fbin` headers or invalid parameters
  (`R == 0`, `L < R`, `alpha < 1.0`);
* `OSError` for filesystem errors (missing input file);
* `RuntimeError` for other build failures.

### CLI: `build_vamana_for_laser`

```bash
./build/tools/build_vamana_index/build_vamana_index \
    --data_path /path/to/base.fbin \
    --index_path_prefix /path/to/graph.index \
    -R 64 -L 100 --alpha 1.2 --seed 1234 -T 0 --build_dram_budget 32.0
```

The CLI's flag surface is unchanged from before the integration change
(`port-diskann-vamana`); it remains the reference trajectory for
alignment research against DiskANN's `search_memory_index`.

### `[build_vamana]` TOML schema

```toml
[build_vamana]             # all fields optional; defaults from kDefaultVamanaBuildParams
L = 200
alpha = 1.2
seed = 1234
num_threads = 0            # 0 → omp_get_num_procs() at call time
dram_budget_gb = 32.0      # single-shard budget; > estimate → partition+merge
# R is NOT listed here — read from [dataset].degree (three-way contract).
```

`R` is sourced exclusively from `[dataset].degree` to keep the build R,
the written `max_observed_degree`, and Laser's `degree_bound` in lockstep.
Setting `[build_vamana].R` raises a config-load error.

### Idempotence

`step_vamana` is skip-if-exists. On each invocation it:

1. resolves the target path (`[paths].vamana` if present, else the
   derived path `{output}/data/{name}/vamana/graph.index`);
2. if the file exists, parses the 24-byte DiskANN header and skips the
   build when `max_observed_degree == [dataset].degree` and
   `frozen_pts == 0`;
3. otherwise rebuilds, overwriting any stale file (with a warning).

This preserves the external-DiskANN trajectory (`gist_diskann.toml`)
and makes a second `all` invocation nearly free.

### Native ±0.3pp variance (multi-threaded)

The single-shard Vamana build is inherently non-deterministic at
`num_threads > 1`: OpenMP's `schedule(dynamic)` assigns nodes to threads
in a wall-clock-dependent order, so the candidate pool ordering inside
`search_for_point_and_prune` is not run-reproducible. Recall at `R=64
L=200 α=1.2` drifts by up to ±0.3pp run-to-run on the same data; this
is the builder's native variance, not a regression.

`num_threads=1` collapses this axis and is what Gate G1 pins.

### First-run cost

`examples/laser/main.py all` on `gist.toml` takes ~15–30 minutes longer
on the first invocation, because the Vamana build no longer has to be
pre-computed out-of-band. Subsequent runs hit the idempotence check and
add a handful of milliseconds.

## Reproducibility

The port ships two RNG modes:

**Normal mode (default)** — behaves exactly like upstream Laser: PCA
sample selection, medoid k-means, and the FHT rotator all run from
unseeded state (`numpy` global RNG, unseeded faiss IVF, and
`std::random_device` respectively). Output is NOT byte-reproducible
across runs; use this when you just need a functioning index.

**Alignment mode** — opt-in via explicit TOML keys. Every RNG is pinned
from config so two runs with the same seeds produce byte-identical
`dsqg_*.index`, `pca.bin`, `medoids`, and `pca_base.fbin`:

| TOML key | Default | Effect when set |
|---|---|---|
| `pca_seed` | `None` (numpy global RNG) | `alayalite.laser.pca.sample_vectors_from_fbin` uses this seed |
| `medoid_seed` | `None` (unseeded) | Both `np.random.choice` and faiss IVF k-means seeded |
| `rotator_seed` | `0` → `std::random_device` | `std::mt19937_64(seed)` drives the Bernoulli sign vector |
| `force_single_thread` | `false` | Pins OMP / BLAS / faiss to 1 thread; requires `build_threads = 1` and all three seeds set |
| `dump_rotator` | `false` | Writes `dsqg_{name}_rotator_signs.bin` alongside the index (for SHA-256 comparison). Requires `rotator_seed ≠ 0`. |

See `examples/laser/configs/synth_20k_768d_alayaP.toml` for a complete
alignment-mode TOML. The harness that drives cross-repo byte-equality
comparison lives at `scripts/laser_alignment/tier_a_byte_equal.py`.

Distributional properties (uniform Bernoulli signs, uniform
training-sample selection) are preserved in both modes; only the RNG
state differs.

## Known Issues

### Preserved upstream page-layout bug

`include/index/graph/laser/qg/qg_builder.hpp` (originally line 256 in
`Laser/symqglib/qg/qg_builder.hpp`) writes pages as
`i * page_size_ + kSectorLen` but `qg.hpp` reads them as
`page_size_ * (id/npp) + (id%npp) * node_len_`. The two paths agree iff
`node_per_page_ == 1`, which holds for all paper datasets with
`main_dim=256` and raw `dim ≥ 768` (gist 960d, bigcode 768d, cohere
1024d, synth 512d). On SIFT-1M (main_dim=64, dim=128) `node_per_page_>1`
and the write/read mismatch collapses recall to ~0.08% — **do not use
this port on SIFT-1M** until the fix lands.

The port preserves the bug verbatim so the byte-equality gate against
baseline Laser outputs still passes. A follow-up change will fix it with
SIFT-1M as the acceptance dataset. See the TODO at the bug site:

```
include/index/graph/laser/qg/qg_builder.hpp
  // TODO(port-laser-disk-index followup): write/read page-layout mismatch;
  // see proposal D8
```

and proposal D8 in `openspec/changes/port-laser-disk-index/proposal.md`.

## How the port relates to AlayaLite's existing QG

AlayaLite already ships a separate `QGBuilder` under
`include/index/graph/qg/qg_builder.hpp`, backed by `space/rabitq_space.hpp`.
That is a distinct in-memory builder with a different quantization flow
(`RabitQSpace` + `DistanceSpaceType` templates). The Laser port is
namespaced `alaya::laser::` and lives under `include/index/graph/laser/`,
so the two coexist cleanly. Which one is "right" for a given workload is
a profile-driven decision outside the scope of this port.
