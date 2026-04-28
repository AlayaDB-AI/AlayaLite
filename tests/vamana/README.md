# Vamana test gates

Python tests, fixtures, and Vamana-specific C++ harnesses live in this
folder.

## Tier-by-tier overview

| Tier | Driver | What it asserts | Cadence | DiskANN side |
|------|--------|-----------------|---------|--------------|
| Tier 0 (CLI byte-equal, AlayaLite-self) | `test_cli_byte_equality.py` | AlayaLite single-shard CLI output is byte-stable to the committed fixture at `seed=1234, T=1`. | Per-commit | n/a |
| Tier A (sharded partition byte-equal vs patched DiskANN) | `test_sharded_byte_equality.py` | `_medoids.bin` SHA matches between AlayaLite and patched DiskANN at matched seeds; structural parity on `num_parts`, header fields, file sizes. | Per-commit | **patched** (`align-diskann-sharded-with-alaya` branch) |
| Tier B (sharded statistical envelope vs unpatched DiskANN) | `test_vamana_alignment` (C++) with `--force_partition --expected_num_parts_envelope <lo> <hi>` | AlayaLite's `num_parts` lands in the empirical envelope of ≥ 3 unpatched DiskANN reruns; recall@10 within 1.0 pp (deterministic refs) or 1.5 pp (unseeded). | **Nightly / manual only** (not registered in CTest) | unpatched |

## Tier A — `test_sharded_byte_equality.py`

Pinned at `R=64 L=100 alpha=1.2 seed=1234 T=1 build_dram_budget=0.05 GiB
sampling_rate=auto` on `synth_100k_512d`. Both pipelines land
`num_parts = 13`. Wall time ≈ 7 minutes (Alaya 3 min + DiskANN 3-4 min)
on a single-socket machine.

The patched DiskANN binary lives on the
`align-diskann-sharded-with-alaya` branch. Build it once with:

```bash
cmake -DUNIT_TEST=ON -B /md1/huangliang/alaya-dev/Laser/DiskANN/build \
                    -S /md1/huangliang/alaya-dev/Laser/DiskANN
cmake --build /md1/huangliang/alaya-dev/Laser/DiskANN/build \
      --target build_merged_vamana_standalone -j
```

Override the binary path with `DISKANN_ALIGNED_BIN=/path/to/binary` if
the default search location fails. The test skips (warns, not fails)
when the binary is missing so the repo remains buildable without the
patched DiskANN side.

### Extended GIST-1M scenario

```bash
pytest -m extended tests/vamana/test_sharded_byte_equality.py
```

Run before each archive of an alignment-related change. The fixture
`fixtures/sharded_gist1m_M0p5.json` is checked in once the Tier A
invariants hold on GIST-1M; subsequent runs assert the AlayaLite
medoids SHA match the fixture.

## Tier B — `test_vamana_alignment` (C++ harness)

Compares AlayaLite's `_mem.index` (single-shard or partition-merged)
to a separately-built DiskANN reference. Two num_parts modes:

* **Strict**: `--expected_num_parts <N>` — used when the DiskANN
  reference is deterministic (e.g. patched DiskANN at matched seeds).
* **Envelope**: `--expected_num_parts_envelope <lo> <hi>` — used when
  the DiskANN reference is unseeded; bounds derived from ≥ 3 reruns.

### GIST-1M envelope (recorded in change `align-diskann-sharded-with-upstream`)

Empirical envelope at `R=64 L=100 -M 0.5 -T 1`, sampling_rate=auto,
captured 2026-04-25 from 3 unpatched DiskANN reruns:

| Run | num_parts | Wall time |
|-----|-----------|-----------|
| 1 | 41 | 51 min |
| 2 | 43 | 51 min |
| 3 | 49 | 44 min |

**Envelope: `--expected_num_parts_envelope 41 49`** (AlayaLite's
seeded `num_parts=43` falls within). Evidence at
`/md1/huangliang/alaya-dev/data/build_graph/diskann_sharded_alignment/tier_a_gist1m_20260425/SUMMARY.md`.

To re-measure:

```bash
# Repeat 3× with fresh OMP runs of unpatched DiskANN build_disk_index;
# record num_parts each time and pass the [min, max] as the envelope.
for i in 1 2 3; do
  rm -rf /tmp/gist1m_unpatched_$i && mkdir -p /tmp/gist1m_unpatched_$i
  /path/to/DiskANN/build/apps/build_disk_index \
    --data_type float --dist_fn l2 \
    --data_path /md1/huangliang/alaya-dev/data/gist1m/base.fbin \
    --index_path_prefix /tmp/gist1m_unpatched_$i/idx \
    -R 64 -L 100 -B 1 -M 0.5 -T 1
  # num_parts visible in the build log line
  # "Saving global k-center pivots" → preceded by "With <N> parts, ..."
done
```

Then:

```bash
./build/tests/vamana/test_vamana_alignment \
  --dataset gist1m --force_partition \
  --expected_num_parts_envelope <lo> <hi> \
  --recall_delta_pp 1.0
```

Re-measurement updates the envelope entry above. Callers MAY also
pass `--expected_num_parts <N>` (strict equality) when the DiskANN
reference is itself patched/seeded.

## Tier 0 — `test_cli_byte_equality.py`

Single-shard CLI determinism guard. Compares `build_vamana_index --data
<synth> --build_dram_budget 1.0` (single-shard path) against the
checked-in fixture. Detects regressions in `VamanaBuilder`'s
single-threaded behavior and `save_graph`'s serialization.
