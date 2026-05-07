# LASER cross-bench

Compare upstream Laser vs AlayaLite Laser on the same dataset and the same
Vamana graphs. All paths and hyperparameters live in a single TOML config —
no source edits needed to change dataset, paths, or sweep grids.

## Layout

```
experiments/laser_cross_bench/
├── _config.py                       # CrossBenchConfig dataclass + TOML loader
├── configs/
│   └── gist1m.toml                  # example config (copy & edit for new datasets)
├── step1_build_vamana_diskann.py    # builder='diskann'  → wraps DiskANN binary
├── step2_build_vamana_alayalite.py  # builder='alayalite' → uses alayalite.vamana
├── step3_bench_original_laser.py    # orig pipeline (PCA → medoid → index → search)
├── step4_bench_alayalite_laser.py   # AlayaLite Index.fit + search
├── step5_compare.py                 # aggregate CSVs into markdown table
├── plot_recall_qps.py               # auto-discover CSVs, plot recall vs QPS
└── run_all.py                       # top-level driver (idempotent, --skip-* flags)
```

Each cell is one combination of `(laser ∈ {orig, lite}, vamana_tag ∈ config.vamana_sources)`
and produces one CSV at `<tmp_dir>/results/{orig,lite}_<tag>.csv`.

## Quickstart

```bash
# 1. Edit configs/gist1m.toml to point at your data_dir / tmp_dir / venvs.
# 2. One-shot run (~30 min for GIST-1M, 6 cells):
python run_all.py --config configs/gist1m.toml

# Subset / iteration:
python run_all.py --config configs/gist1m.toml --tags diskann --laser-only lite
python run_all.py --config configs/gist1m.toml --force-bench    # rerun bench, keep vamana
python plot_recall_qps.py --config configs/gist1m.toml --tags alayalite_l100 alayalite_l200
```

Each step also runs standalone, e.g.:

```bash
python step1_build_vamana_diskann.py  --config configs/gist1m.toml --vamana-tag diskann
python step2_build_vamana_alayalite.py --config configs/gist1m.toml --vamana-tag alayalite_l100
/path/to/Laser/.venv/bin/python step3_bench_original_laser.py \
    --config configs/gist1m.toml --vamana-tag diskann
python step4_bench_alayalite_laser.py --config configs/gist1m.toml --vamana-tag diskann
```

## New dataset

```bash
cp configs/gist1m.toml configs/sift1m.toml
# Edit:
#   name           → "sift1m"
#   paths.data_dir → /path/to/sift1m
#   dataset.prefix → "sift"   (assumes sift_base.fbin / sift_query.fbin / sift_gt.ibin)
#   build.main_dim → 128 (SIFT is 128d, no PCA reduction)
python run_all.py --config configs/sift1m.toml
```

Override the `<prefix>_{base,query,gt}.{fbin,ibin}` convention with explicit
`base_filename` / `query_filename` / `gt_filename` in `[dataset]`.

## TOML schema

See `configs/gist1m.toml`. Five top-level sections:

| Section | Purpose |
|---------|---------|
| `paths` | `data_dir`, `tmp_dir`, optional `laser_venv` (step3) and `diskann_binary` (step1) |
| `dataset` | `prefix` (auto file naming) + optional explicit filenames |
| `build` | LASER `BuildParams` (R / main_dim / L / alpha / ef_indexing / ep_num) |
| `vamana_sources` | array of tables, each `{tag, builder, L}` |
| `bench` | `k`, `beam`, `warmup`, `runs`, `ef_sweep`, `build_threads`, `seed`, ... |

Optional fields (`laser_venv`, `diskann_binary`) raise a precise error
only when the step that needs them is invoked, so you can run an
AlayaLite-only or no-DiskANN subset without filling them in.

## Why TOML, not YAML

Both AlayaLite venv and the upstream Laser venv must read this file. TOML is
in the Python stdlib (`tomllib`, 3.11+); YAML is not. No external dependency.
