# Disk LASER Fixture

`build_laser_fixture.py` generates the deterministic LASER artifact set used by
the disk LASER adapter tests.

Default contract:

- `count = 2048`
- `dim = 128`
- `R = 64`
- `seed = 42`
- `prefix = dsqg_seg_00000001`
- `main_dim = 128`

Default outputs under the selected output directory:

- `dsqg_seg_00000001_input.fbin` - raw input vectors used by recall checks
- `dsqg_seg_00000001_R64_MD128.index` - required native LASER index
- `dsqg_seg_00000001_R64_MD128.index_rotator` - required native rotator sidecar
- `dsqg_seg_00000001_R64_MD128.index_cache_ids` - required native cache-id sidecar
- `dsqg_seg_00000001_R64_MD128.index_cache_nodes` - required native cache-node sidecar
- `dsqg_seg_00000001_pca.bin` - optional PCA params sidecar
- `dsqg_seg_00000001_medoids` - optional medoid vectors sidecar
- `dsqg_seg_00000001_medoids_indices` - optional medoid indices sidecar

The script also leaves build-only intermediates in the fixture directory:
`dsqg_seg_00000001_pca_base.fbin` for `QGBuilder::build` and
`dsqg_seg_00000001_vamana_graph.index` for the temporary Vamana graph.
The LASER segment importer ignores these extra files.

Generation is idempotent. Existing files are skipped when their headers and
sizes match the requested fixture contract. Pass `--no-optional-sidecars` to
produce only the required native LASER artifacts.
