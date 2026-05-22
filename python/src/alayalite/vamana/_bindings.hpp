// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Vamana (alaya::vamana) pybind11 registration.
//
// Exposes `register_vamana_module(py::module_&)` which attaches a single
// function `build_index` to the given pybind module. Called from
// python/src/pybind.cpp to register the `alayalite.vamana` submodule.
//
// Defaults for the non-R parameters are pulled from
// `alaya::vamana::kDefaultVamanaBuildParams` — the one place in the tree
// where Vamana build defaults are literally declared. Per spec, `R` is
// required and has no default.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <string>
#include <system_error>

#include "index/graph/vamana/build_dispatch.hpp"
#include "utils/platform_fs.hpp"  // is_readable_regular_file (portable access/R_OK check)

namespace alaya::vamana::bindings {

namespace py = pybind11;

// build_index — Python entry point. Maps the spec D10 exception surface:
//
//   ValueError  — malformed .fbin header / invalid parameter values
//                 (automatic via pybind11's std::invalid_argument mapping)
//   OSError     — filesystem errors (raised here before entering C++)
//   RuntimeError— all other build failures
//                 (automatic via pybind11's std::runtime_error mapping)
//
// Paths are std::string, not numpy arrays, so pybind11's type check raises
// TypeError when a user passes an ndarray — satisfying the spec scenario
// "numpy input rejected".
inline void build_index(const std::string &data_path,
                        const std::string &output_path,
                        uint32_t R,
                        uint32_t L,
                        float alpha,
                        uint64_t seed,
                        uint32_t num_threads,
                        float dram_budget_gb,
                        float sampling_rate) {
  // Pre-check for OSError semantics. The underlying C++ code would raise
  // std::runtime_error on open() failure (mapping to RuntimeError via
  // pybind11 default), but the spec contract is OSError for unreadable
  // input. Validate at the Python boundary that data_path names a
  // regular file we can read; surface directories, broken symlinks, and
  // permission errors as OSError rather than falling through to a
  // generic RuntimeError later.
  std::error_code ec;
  const auto status = std::filesystem::status(data_path, ec);
  if (ec || !std::filesystem::exists(status)) {
    PyErr_SetString(PyExc_OSError, ("data_path does not exist: " + data_path).c_str());
    throw py::error_already_set();
  }
  if (!std::filesystem::is_regular_file(status)) {
    PyErr_SetString(PyExc_OSError,
                    ("data_path is not a regular file (got " +
                     std::to_string(static_cast<int>(status.type())) + "): " + data_path)
                        .c_str());
    throw py::error_already_set();
  }
  // R_OK check catches permission errors that is_regular_file won't.
  if (!::alaya::platform::is_readable_regular_file(data_path)) {
    PyErr_SetString(PyExc_OSError, ("data_path is not readable: " + data_path).c_str());
    throw py::error_already_set();
  }

  // The BuildVamanaParams struct stores paths as string_view. The C++
  // build_vamana call returns before this function frame is destroyed,
  // so the string_views remain valid throughout the build.
  BuildVamanaParams params = kDefaultVamanaBuildParams;
  params.data_path = data_path;
  params.output_path = output_path;
  params.R = R;
  params.L = L;
  params.alpha = alpha;
  params.seed = seed;
  params.num_threads = num_threads;
  params.build_dram_budget_gb = dram_budget_gb;
  params.sampling_rate = sampling_rate;

  build_vamana(params);
}

inline void register_vamana_module(py::module_ &m) {
  m.doc() = R"pbdoc(
AlayaLite Vamana graph builder (DiskANN-format .index output).

This module exposes a single function `build_index` that wraps the
`alaya::vamana::build_vamana` dispatch library. The output is a DiskANN
single-file `.index` binary, directly consumable by `alayalite.laser.Index.build_index`.
)pbdoc";

  // Defaults for all non-R parameters reference kDefaultVamanaBuildParams
  // — no duplicate literals. R is intentionally declared without a
  // default so callers pass it explicitly (three-way R contract:
  // Vamana build R = Laser degree_bound = written max_observed_degree).
  m.def("build_index",
        &build_index,
        py::arg("data_path"),
        py::arg("output_path"),
        py::arg("R"),
        py::arg("L") = kDefaultVamanaBuildParams.L,
        py::arg("alpha") = kDefaultVamanaBuildParams.alpha,
        py::arg("seed") = kDefaultVamanaBuildParams.seed,
        py::arg("num_threads") = kDefaultVamanaBuildParams.num_threads,
        py::arg("dram_budget_gb") = kDefaultVamanaBuildParams.build_dram_budget_gb,
        py::arg("sampling_rate") = kDefaultVamanaBuildParams.sampling_rate,
        R"pbdoc(
Build a Vamana graph and write a DiskANN-format .index file.

Parameters
----------
data_path : str
    Input .fbin file (DiskANN float32 layout: uint32 num, uint32 dim, data).
output_path : str
    Output .index path. Parent directories are created if missing.
R : int
    Graph degree bound (required, no default). Must equal the degree the
    Laser QG index will consume (see `[dataset].degree` in the TOML
    config); mismatch triggers the `max_observed_degree == degree_bound`
    assert in QGBuilder.
L : int
    Build-time beam width (default 100). Must be >= R.
alpha : float
    α-RNG pruning parameter (default 1.2). Must be >= 1.0.
seed : int
    RNG seed (default 1234). Applied to k-means init, shard shuffles, and
    the OpenMP-independent parts of the Vamana build.
num_threads : int
    OpenMP thread count (default 0 → omp_get_num_procs() at call time).
dram_budget_gb : float
    Single-shard RAM budget in GiB (default 32.0). When the estimated
    single-shard RAM exceeds this, the builder switches to a k-means
    partition + union-shuffle-cut merge path.

Raises
------
ValueError
    Malformed .fbin header or invalid parameter values.
OSError
    Input file missing or unreadable.
RuntimeError
    All other build failures.
)pbdoc");
}

}  // namespace alaya::vamana::bindings
