# Pybind Binding Layout

`python/src/pybind.cpp` is the only root-level entrypoint TU. It wires top-level registrations and
submodule registrations into `PYBIND11_MODULE`.

`python/src/bindings/` contains top-level binding registrations and helper method groups for
`_alayalitepy`.

`python/src/submodules/` contains thin registration shells for dedicated Python submodules.

## Ownership Table

| Symbol | Python Path | Registration TU |
| --- | --- | --- |
| `Client` | `alayalite.Client` | `python/src/bindings/pybind_index.cpp` |
| `IndexParams` | `alayalite.IndexParams` | `python/src/bindings/pybind_index.cpp` |
| `PyIndexInterface` | `alayalite.PyIndexInterface` | `python/src/bindings/pybind_index.cpp` |
| `DiskCollection` | `alayalite.DiskCollection` | `python/src/submodules/pybind_disk.cpp` |
| `build_index` | `alayalite.vamana.build_index` | `python/src/submodules/pybind_vamana.cpp` |
| `Index` | `alayalite.laser.Index` | `python/src/submodules/pybind_laser.cpp` |
