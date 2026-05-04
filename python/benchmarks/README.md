# Python benchmark harnesses

This directory houses thin wrappers around the canonical Python
benchmark CLIs that live in `python/src/alayalite/bench/`.

## Available CLIs

- `python -m alayalite.bench.disk_collection` — the unified
  DiskCollection sweep harness (engines: `disk_flat`, `disk_vamana`,
  `disk_laser`). See OpenSpec capability `disk-collection-benchmark`.
- `python -m alayalite.bench.laser_compare` — paired native
  `alayalite.laser.Index` vs
  `DiskCollection(index_type="disk_laser")` harness; loads one
  precomputed LASER artifact directory and emits a `comparison` JSON
  block quantifying the wrapper-layer adapter overhead. See OpenSpec
  capability
  [`laser-native-equivalence-benchmark`](../../openspec/specs/laser-native-equivalence-benchmark/spec.md)
  and archived change
  [`python-laser-native-equivalence-benchmark`](../../openspec/changes/archive/2026-05-02-python-laser-native-equivalence-benchmark/).

The paired harness is gated on the runtime probe
`alayalite.bench._engines.probe_disk_laser_supported`; on builds without
`ALAYA_ENABLE_LASER=ON` it prints a single skip line and exits 0
without writing any output.

The smoke test for `laser_compare` is
`python/tests/test_bench_laser_compare_smoke.py`; it is collected
as `skipped` on unsupported builds and runs end-to-end on the
deterministic 256-row / 128-dim fixture from
`python/tests/fixtures/laser/builder.py` on supported builds.

## Notes

- The legacy `disk_laser_smoke.py` script in this directory is a
  compatibility wrapper preserving the historical stdout contract;
  new work should target the canonical CLIs above.
- Large external datasets are off by default. The unified
  `disk_collection` harness opts in via `--dataset-root <PATH>`; the
  paired `laser_compare` harness takes its inputs through
  `--laser-src-dir`, `--vectors`, `--queries-path`, and
  `--ground-truth` flags directly. CI does not exercise large
  datasets — the `laser_compare` smoke test is the only LASER paired
  path exercised on every build.
