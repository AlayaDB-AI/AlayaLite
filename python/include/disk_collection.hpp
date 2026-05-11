// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include "index/disk/disk_collection.hpp"
#include "index/disk/segment_factory.hpp"
#include "index/disk/types.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk::pybindings {

namespace py = pybind11;

// Pre-gate the engine via `engine_supported_v1` and re-raise the C++ factory's
// own `std::runtime_error` ("DiskSegmentFactory: engine '<engine>' not
// implemented in v1") as `py::value_error`. This preserves the spec contract
// "the message SHALL be sourced from the C++ factory's throw site (single
// source of truth)" while avoiding fragile substring matching against
// generic `std::runtime_error` paths in the C++ ctor / open() call stacks.
inline auto assert_engine_supported_at_binding(DiskIndexType type) -> void {
  if (!engine_supported_v1(type)) {
    try {
      detail::throw_unsupported_engine(type);
    } catch (const std::runtime_error &e) {
      throw py::value_error(e.what());
    }
  }
}

inline auto is_unsupported_engine_runtime_error(const std::string &msg) -> bool {
  return msg.find("not implemented in v1") != std::string::npos &&
         (msg.find("disk_laser") != std::string::npos ||
          msg.find("disk_vamana") != std::string::npos ||
          msg.find("disk_flat") != std::string::npos);
}

inline auto index_type_from_string_strict(const std::string &s) -> DiskIndexType {
  if (s == "disk_flat") {
    return DiskIndexType::Flat;
  }
  if (s == "disk_vamana") {
    return DiskIndexType::Vamana;
  }
  if (s == "disk_laser") {
    // No hard binding-policy veto: forward the parsed enum and let
    // `assert_engine_supported_at_binding` (called from the ctor / open)
    // re-raise the C++ factory's "not implemented in v1" runtime_error as
    // ValueError on unsupported builds. Single source of truth lives in
    // `segment_factory.hpp::throw_unsupported_engine`.
    return DiskIndexType::Laser;
  }
  std::string supported = "\"disk_flat\" and \"disk_vamana\"";
  if constexpr (engine_supported_v1(DiskIndexType::Laser)) {
    supported = "\"disk_flat\", \"disk_vamana\", and \"disk_laser\"";
  }
  throw py::value_error("DiskCollection: unknown index_type \"" + s + "\"; supported values are " +
                        supported);
}

inline auto metric_name(MetricType metric) -> std::string {
  switch (metric) {
    case MetricType::L2:
      return "L2";
    case MetricType::IP:
      return "IP";
    case MetricType::COS:
      return "COS";
    case MetricType::NONE:
      return "NONE";
  }
  return "unknown";
}

inline auto is_finite_f64(double value) -> bool {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(value));
  return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

inline auto is_finite_f32(float value) -> bool {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(value));
  return (bits & 0x7F800000U) != 0x7F800000U;
}

inline auto resolve_vamana_num_threads(int64_t num_threads) -> int64_t {
  if (num_threads != 0) {
    return num_threads;
  }
  const char *env = std::getenv("OMP_NUM_THREADS");
  if (env == nullptr || *env == '\0') {
    return 0;
  }
  try {
    size_t pos = 0;
    const auto value = std::stoll(env, &pos);
    if (pos == std::strlen(env) && value > 0 &&
        value <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return value;
    }
  } catch (const std::exception &) {
    // Invalid environment values keep the C++ adapter default.
  }
  return 0;
}

// Resolves `num_threads = 0` for batch_search the same way Vamana does, but
// adds a `hardware_concurrency()` fallback (with a 1-thread floor) because the
// C++ `DiskCollection::batch_search` rejects `num_threads = 0` outright.
// Caller must validate `num_threads >= 0` before invoking this helper.
inline auto resolve_batch_num_threads(int64_t num_threads) -> uint32_t {
  if (num_threads > 0) {
    return static_cast<uint32_t>(num_threads);
  }
  const char *env = std::getenv("OMP_NUM_THREADS");
  if (env != nullptr && *env != '\0') {
    try {
      size_t pos = 0;
      const auto value = std::stoll(env, &pos);
      if (pos == std::strlen(env) && value >= 1 &&
          value <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return static_cast<uint32_t>(value);
      }
    } catch (const std::exception &) {
      // Invalid environment values fall through to the hardware concurrency
      // probe; no separate signal is surfaced to the caller.
    }
  }
  const auto hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1U : hw;
}

inline auto validate_vamana_params(int64_t r,
                                   int64_t l,
                                   double alpha,
                                   int64_t seed,
                                   int64_t num_threads) -> VamanaSegmentBuildParams {
  if (r <= 0 || r > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    throw py::value_error("DiskCollection: vamana_R must be in [1, 2**32 - 1]");
  }
  if (l <= 0 || l > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    throw py::value_error("DiskCollection: vamana_L must be in [1, 2**32 - 1]");
  }
  if (l < r) {
    throw py::value_error("DiskCollection: vamana_L must be >= vamana_R");
  }
  if (!is_finite_f64(alpha) || alpha < 1.0) {
    throw py::value_error("DiskCollection: vamana_alpha must be finite and >= 1.0");
  }
  if (alpha > static_cast<double>(std::numeric_limits<float>::max())) {
    throw py::value_error("DiskCollection: vamana_alpha must be representable as finite float32");
  }
  if (seed < 0 || seed > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    throw py::value_error("DiskCollection: vamana_seed must be in [0, 2**32 - 1]");
  }
  const auto resolved_threads = resolve_vamana_num_threads(num_threads);
  if (resolved_threads < 0 ||
      resolved_threads > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    throw py::value_error("DiskCollection: vamana_num_threads must be in [0, 2**32 - 1]");
  }
  VamanaSegmentBuildParams params;
  params.R = static_cast<uint32_t>(r);
  params.L = static_cast<uint32_t>(l);
  params.alpha = static_cast<float>(alpha);
  if (!is_finite_f32(params.alpha)) {
    throw py::value_error("DiskCollection: vamana_alpha must be representable as finite float32");
  }
  params.seed = static_cast<uint32_t>(seed);
  params.num_threads = static_cast<uint32_t>(resolved_threads);
  return params;
}

inline auto require_dtype(const py::array &array,
                          const py::dtype &dtype,
                          const std::string &message) -> void {
  if (!array.dtype().is(dtype)) {
    throw py::type_error(message);
  }
}

class PyDiskCollection {
 public:
  PyDiskCollection(const std::string &path,
                   uint32_t dim,
                   MetricType metric,
                   const std::string &index_type,
                   size_t max_pending_bytes = DiskCollection::kDefaultMaxPendingBytes,
                   int64_t vamana_R = 64,
                   int64_t vamana_L = 200,
                   double vamana_alpha = 1.2,
                   int64_t vamana_seed = 1234,
                   int64_t vamana_num_threads = 0) {
    const auto parsed_index_type = index_type_from_string_strict(index_type);
    VamanaSegmentBuildParams vamana_params;
    if (parsed_index_type == DiskIndexType::Vamana) {
      if (metric != MetricType::L2) {
        throw py::value_error("DiskCollection: disk_vamana metric " + metric_name(metric) +
                              " is not supported; Vamana v1 supports L2 only");
      }
      vamana_params =
          validate_vamana_params(vamana_R, vamana_L, vamana_alpha, vamana_seed, vamana_num_threads);
    }
    assert_engine_supported_at_binding(parsed_index_type);
    impl_ = std::make_unique<DiskCollection>(path,
                                             dim,
                                             metric,
                                             parsed_index_type,
                                             max_pending_bytes,
                                             vamana_params);
    cached_index_type_ = parsed_index_type;
  }

  static auto open(const std::string &path) -> std::shared_ptr<PyDiskCollection> {
    auto inner = [&path]() {
      try {
        return DiskCollection::open(path);
      } catch (const std::runtime_error &e) {
        const std::string msg = e.what();
        if (is_unsupported_engine_runtime_error(msg)) {
          throw py::value_error(msg);
        }
        throw;
      }
    }();
    const auto manifest =
        CollectionManifest::load(std::filesystem::path(path) / "collection_manifest.txt");
    return std::shared_ptr<PyDiskCollection>(
        new PyDiskCollection(std::move(inner), manifest.index_type));
  }

  void add(py::array vectors, py::array ids) {
    // Pre-call reject for disk_laser collections so the binding boundary
    // surfaces a clear "use import_laser_segment" message before the GIL is
    // released. Same dual-substring contract as the C++ `add_batch` reject.
    if (cached_index_type_ == DiskIndexType::Laser) {
      throw std::runtime_error(
          "DiskCollection.add: disk_laser add not implemented in v1; use "
          "import_laser_segment for precomputed LASER artifacts");
    }
    require_dtype(vectors,
                  py::dtype::of<float>(),
                  "DiskCollection.add: vectors.dtype must be float32");
    require_dtype(ids, py::dtype::of<uint64_t>(), "DiskCollection.add: ids.dtype must be uint64");
    if (vectors.ndim() != 2) {
      throw py::value_error("DiskCollection.add: vectors must be 2D (got ndim=" +
                            std::to_string(vectors.ndim()) + ")");
    }
    if (ids.ndim() != 1) {
      throw py::value_error(
          "DiskCollection.add: ids must be 1D (got ndim=" + std::to_string(ids.ndim()) + ")");
    }
    const auto n_rows = static_cast<uint64_t>(vectors.shape(0));
    const auto v_dim = static_cast<uint64_t>(vectors.shape(1));
    const auto n_ids = static_cast<uint64_t>(ids.shape(0));
    if (v_dim != impl_->dim()) {
      throw py::value_error("DiskCollection.add: vectors.shape[1]=" + std::to_string(v_dim) +
                            " does not match collection dim=" + std::to_string(impl_->dim()));
    }
    if (n_ids != n_rows) {
      throw py::value_error("DiskCollection.add: ids.shape[0]=" + std::to_string(n_ids) +
                            " does not match vectors.shape[0]=" + std::to_string(n_rows));
    }
    if ((vectors.flags() & py::array::c_style) == 0) {
      throw py::type_error("DiskCollection.add: vectors must be C-contiguous");
    }
    if ((ids.flags() & py::array::c_style) == 0) {
      throw py::type_error("DiskCollection.add: ids must be C-contiguous");
    }

    const auto *vector_data = static_cast<const float *>(vectors.data());
    const auto *id_data = static_cast<const uint64_t *>(ids.data());
    {
      py::gil_scoped_release release;
      std::unique_lock<std::shared_mutex> lock(mutex_);
      impl_->add_batch(vector_data, id_data, n_rows);
    }
  }

  void flush() {
    // Unconditional reject for disk_laser regardless of pending state. The C++
    // `flush()` is a no-op when pending is empty (Laser collections always
    // have empty pending because `add()` rejects), so silent success would
    // mislead callers who expect `flush()` to materialize a segment. The
    // binding adds an explicit reject pointing at `import_laser_segment` —
    // see design D3 for the rationale; this is the only place where the
    // Python contract intentionally diverges from the C++ semantics.
    if (cached_index_type_ == DiskIndexType::Laser) {
      throw std::runtime_error(
          "DiskCollection.flush: disk_laser flush not implemented in v1; use "
          "import_laser_segment for precomputed LASER artifacts");
    }
    py::gil_scoped_release release;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    impl_->flush();
  }

  void import_laser_segment(py::object src_dir_obj, py::array labels, bool copy) {
    // Validation order matches the spec (`disk-laser-python-binding`
    // `Python import_laser_segment method` requirement):
    //   1. wrapped collection is disk_laser
    //   2. src_dir exists and is a directory
    //   3. labels.dtype == numpy.uint64
    //   4. labels is C-contiguous
    //   5. labels.ndim == 1 and shape[0] >= 1
    //   6. copy is True (v1 rejects copy=False with NotImplementedError)
    // Step 6 runs LAST so `copy=False` does not short-circuit the
    // src_dir / labels error scenarios that the spec parametrizes
    // independently. The "no seg_* directory created" contract is
    // preserved either way because the C++ importer is not invoked
    // until step 7.
    if (cached_index_type_ != DiskIndexType::Laser) {
      throw std::runtime_error(
          "DiskCollection.import_laser_segment: import_laser_segment requires a "
          "disk_laser collection");
    }
    std::string src_dir_str;
    {
      auto fspath = py::module_::import("os").attr("fspath");
      src_dir_str = std::string(py::str(fspath(src_dir_obj)));
    }
    std::filesystem::path src_dir_path(src_dir_str);
    {
      std::error_code ec;
      const bool is_dir = std::filesystem::is_directory(src_dir_path, ec);
      if (ec || !is_dir) {
        throw py::value_error(
            "DiskCollection.import_laser_segment: src_dir is not an existing directory: " +
            src_dir_str);
      }
    }
    require_dtype(labels,
                  py::dtype::of<uint64_t>(),
                  "DiskCollection.import_laser_segment: labels.dtype must be uint64");
    if ((labels.flags() & py::array::c_style) == 0) {
      throw py::type_error("DiskCollection.import_laser_segment: labels must be C-contiguous");
    }
    if (labels.ndim() != 1) {
      throw py::value_error("DiskCollection.import_laser_segment: labels must be 1D (got ndim=" +
                            std::to_string(labels.ndim()) + ")");
    }
    const auto n = static_cast<uint64_t>(labels.shape(0));
    if (n == 0) {
      throw py::value_error("DiskCollection.import_laser_segment: labels must be non-empty");
    }
    if (!copy) {
      PyErr_SetString(PyExc_NotImplementedError,
                      "DiskCollection.import_laser_segment: v1 supports copy=True only; "
                      "hard-link import requires a future C++ API extension to "
                      "DiskCollection::import_laser_segment to accept "
                      "LaserSegmentImportParams");
      throw py::error_already_set();
    }
    const auto *labels_data = static_cast<const uint64_t *>(labels.data());
    {
      py::gil_scoped_release release;
      std::unique_lock<std::shared_mutex> lock(mutex_);
      impl_->import_laser_segment(src_dir_path, labels_data, n);
    }
  }

  auto search(py::array query, int k, int ef, int64_t beam_width)
      -> std::vector<std::tuple<uint64_t, float>> {
    if (k <= 0) {
      throw py::value_error("DiskCollection.search: k must be > 0 (got " + std::to_string(k) + ")");
    }
    if (ef <= 0) {
      throw py::value_error("DiskCollection.search: ef must be > 0 (got " + std::to_string(ef) +
                            ")");
    }
    if (beam_width <= 0 ||
        beam_width > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      throw py::value_error("DiskCollection.search: beam_width must be in [1, 2**32 - 1] (got " +
                            std::to_string(beam_width) + ")");
    }
    require_dtype(query,
                  py::dtype::of<float>(),
                  "DiskCollection.search: query.dtype must be float32");
    if (query.ndim() != 1) {
      throw py::value_error("DiskCollection.search: query must be 1D (got ndim=" +
                            std::to_string(query.ndim()) + ")");
    }
    if (static_cast<uint64_t>(query.shape(0)) != impl_->dim()) {
      throw py::value_error(
          "DiskCollection.search: query.shape[0]=" + std::to_string(query.shape(0)) +
          " does not match collection dim=" + std::to_string(impl_->dim()));
    }
    if ((query.flags() & py::array::c_style) == 0) {
      throw py::type_error("DiskCollection.search: query must be C-contiguous");
    }

    // Engine-uniform NaN / Inf check on `query`. Uses the bit-pattern helper
    // because `-Ofast` / `-ffast-math` folds `<cmath>` finite checks into
    // constants (per `project_ofast_finiteness_check`). Applies before
    // engine dispatch so disk_flat / disk_vamana / disk_laser see the same
    // contract.
    const auto *query_data = static_cast<const float *>(query.data());
    const uint64_t dim = impl_->dim();
    for (uint64_t i = 0; i < dim; ++i) {
      const float v = query_data[i];
      if (!is_finite_f32(v)) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(v));
        const bool nan_bit = (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0;
        const bool sign_bit = (bits & 0x80000000U) != 0;
        const std::string kind = nan_bit ? "nan" : (sign_bit ? "-inf" : "inf");
        throw py::value_error("DiskCollection.search: query[" + std::to_string(i) + "] is " + kind +
                              " (non-finite query components are not allowed)");
      }
    }

    DiskSearchOptions opts;
    opts.top_k = static_cast<uint32_t>(k);
    opts.ef = static_cast<uint32_t>(ef);
    opts.beam_width = static_cast<uint32_t>(beam_width);

    std::vector<DiskSearchHit> hits;
    {
      py::gil_scoped_release release;
      std::shared_lock<std::shared_mutex> lock(mutex_);
      hits = impl_->search(query_data, opts);
    }

    std::vector<std::tuple<uint64_t, float>> out;
    out.reserve(hits.size());
    for (const auto &h : hits) {
      out.emplace_back(h.label, h.distance);
    }
    return out;
  }

  auto batch_search(py::array queries, int k, int ef, int64_t beam_width, int64_t num_threads)
      -> py::array {
    const auto prep = prepare_batch_search(queries, k, ef, beam_width, num_threads, "batch_search");
    py::array_t<uint64_t> labels({static_cast<py::ssize_t>(prep.n_queries),
                                  static_cast<py::ssize_t>(static_cast<uint32_t>(k))});
    auto *labels_ptr = static_cast<uint64_t *>(labels.request().ptr);
    std::fill_n(labels_ptr,
                prep.n_queries * static_cast<uint64_t>(k),
                std::numeric_limits<uint64_t>::max());
    {
      py::gil_scoped_release release;
      std::shared_lock<std::shared_mutex> lock(mutex_);
      impl_->batch_search(prep.queries_data,
                          prep.n_queries,
                          prep.opts,
                          prep.resolved_threads,
                          labels_ptr,
                          /*out_distances=*/nullptr);
    }
    return labels;
  }

  auto batch_search_with_distance(py::array queries,
                                  int k,
                                  int ef,
                                  int64_t beam_width,
                                  int64_t num_threads) -> py::tuple {
    const auto prep =
        prepare_batch_search(queries, k, ef, beam_width, num_threads, "batch_search_with_distance");
    py::array_t<uint64_t> labels({static_cast<py::ssize_t>(prep.n_queries),
                                  static_cast<py::ssize_t>(static_cast<uint32_t>(k))});
    py::array_t<float> distances({static_cast<py::ssize_t>(prep.n_queries),
                                  static_cast<py::ssize_t>(static_cast<uint32_t>(k))});
    auto *labels_ptr = static_cast<uint64_t *>(labels.request().ptr);
    auto *distances_ptr = static_cast<float *>(distances.request().ptr);
    const auto total = prep.n_queries * static_cast<uint64_t>(k);
    std::fill_n(labels_ptr, total, std::numeric_limits<uint64_t>::max());
    std::fill_n(distances_ptr, total, std::numeric_limits<float>::quiet_NaN());
    {
      py::gil_scoped_release release;
      std::shared_lock<std::shared_mutex> lock(mutex_);
      impl_->batch_search(prep.queries_data,
                          prep.n_queries,
                          prep.opts,
                          prep.resolved_threads,
                          labels_ptr,
                          distances_ptr);
    }
    return py::make_tuple(labels, distances);
  }

  auto size() const -> uint64_t {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return impl_->size();
  }
  auto dim() const -> uint32_t {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return impl_->dim();
  }

 private:
  PyDiskCollection(DiskCollection &&inner, DiskIndexType index_type)
      : impl_(std::make_unique<DiskCollection>(std::move(inner))), cached_index_type_(index_type) {}

  // Result of `prepare_batch_search`. The `queries_data` pointer aliases into
  // the caller-supplied py::array, which the public methods keep alive for
  // the duration of the C++ call (the array is a function parameter).
  struct BatchPrep {
    uint64_t n_queries;
    uint32_t resolved_threads;
    DiskSearchOptions opts;
    const float *queries_data;
  };

  // Common validation prologue shared by `batch_search` and
  // `batch_search_with_distance`. Runs the dtype / shape / contiguity / k /
  // ef / beam_width / num_threads / per-element finite checks (all spec
  // requirement points 1..9) and resolves `num_threads = 0` via
  // `resolve_batch_num_threads`. The GIL is held throughout: per spec the
  // first validation failure must raise without entering the C++ batch path.
  auto prepare_batch_search(const py::array &queries,
                            int k,
                            int ef,
                            int64_t beam_width,
                            int64_t num_threads,
                            const std::string &method) const -> BatchPrep {
    require_dtype(queries,
                  py::dtype::of<float>(),
                  "DiskCollection." + method + ": queries.dtype must be float32");
    if (queries.ndim() != 2) {
      throw py::value_error("DiskCollection." + method + ": queries must be 2D (got ndim=" +
                            std::to_string(queries.ndim()) + ")");
    }
    if (static_cast<uint64_t>(queries.shape(1)) != impl_->dim()) {
      throw py::value_error("DiskCollection." + method +
                            ": queries.shape[1]=" + std::to_string(queries.shape(1)) +
                            " does not match collection dim=" + std::to_string(impl_->dim()));
    }
    if ((queries.flags() & py::array::c_style) == 0) {
      throw py::type_error("DiskCollection." + method + ": queries must be C-contiguous");
    }
    if (k <= 0) {
      throw py::value_error("DiskCollection." + method + ": k must be > 0 (got " +
                            std::to_string(k) + ")");
    }
    if (ef <= 0) {
      throw py::value_error("DiskCollection." + method + ": ef must be > 0 (got " +
                            std::to_string(ef) + ")");
    }
    if (beam_width <= 0 ||
        beam_width > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      throw py::value_error("DiskCollection." + method +
                            ": beam_width must be in [1, 2**32 - 1] (got " +
                            std::to_string(beam_width) + ")");
    }
    if (num_threads < 0 ||
        num_threads > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      throw py::value_error("DiskCollection." + method +
                            ": num_threads must be in [0, 2**32 - 1] (got " +
                            std::to_string(num_threads) + ")");
    }

    const auto n_queries = static_cast<uint64_t>(queries.shape(0));
    const auto dim = impl_->dim();
    const auto *queries_data = static_cast<const float *>(queries.data());

    // Per-element finite check. Bit-pattern helper because `-Ofast` /
    // `-ffast-math` folds `<cmath>` finite checks (per
    // `project_ofast_finiteness_check`). Engine-uniform: applies before
    // engine dispatch so disk_flat / disk_vamana / disk_laser all see the
    // same contract.
    for (uint64_t row = 0; row < n_queries; ++row) {
      for (uint64_t col = 0; col < dim; ++col) {
        const float v = queries_data[row * dim + col];
        if (!is_finite_f32(v)) {
          uint32_t bits = 0;
          std::memcpy(&bits, &v, sizeof(v));
          const bool nan_bit = (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0;
          const bool sign_bit = (bits & 0x80000000U) != 0;
          const std::string kind = nan_bit ? "nan" : (sign_bit ? "-inf" : "inf");
          throw py::value_error("DiskCollection." + method + ": queries[" + std::to_string(row) +
                                "][" + std::to_string(col) + "] is " + kind +
                                " (non-finite query components are not allowed)");
        }
      }
    }

    BatchPrep prep;
    prep.n_queries = n_queries;
    prep.resolved_threads = resolve_batch_num_threads(num_threads);
    prep.opts.top_k = static_cast<uint32_t>(k);
    prep.opts.ef = static_cast<uint32_t>(ef);
    prep.opts.beam_width = static_cast<uint32_t>(beam_width);
    prep.queries_data = queries_data;
    return prep;
  }

  std::unique_ptr<DiskCollection> impl_;
  mutable std::shared_mutex mutex_;
  // Cached at construction so add() / flush() / import_laser_segment() reject
  // paths can fire without reading the C++ manifest on every call.
  DiskIndexType cached_index_type_{DiskIndexType::Flat};
};

inline void register_disk_collection(py::module_ &m) {
  py::class_<PyDiskCollection, std::shared_ptr<PyDiskCollection>>(m, "DiskCollection")
      .def(py::init<const std::string &,
                    uint32_t,
                    MetricType,
                    const std::string &,
                    size_t,
                    int64_t,
                    int64_t,
                    double,
                    int64_t,
                    int64_t>(),
           py::arg("path"),
           py::arg("dim"),
           py::arg("metric"),
           py::arg("index_type"),
           py::kw_only(),
           py::arg("max_pending_bytes") = DiskCollection::kDefaultMaxPendingBytes,
           py::arg("vamana_R") = 64,
           py::arg("vamana_L") = 200,
           py::arg("vamana_alpha") = 1.2,
           py::arg("vamana_seed") = 1234,
           py::arg("vamana_num_threads") = 0)
      .def_static("open", &PyDiskCollection::open, py::arg("path"))
      .def("add", &PyDiskCollection::add, py::arg("vectors"), py::arg("ids"))
      .def("flush",
           &PyDiskCollection::flush,
           "Materialize any pending vectors as a new on-disk segment.\n\n"
           "On a `disk_laser` collection this method always raises\n"
           "`RuntimeError` with the dual-substring `disk_laser` /\n"
           "`not implemented in v1` contract, regardless of pending state.\n"
           "This intentionally diverges from the underlying C++ `flush()`,\n"
           "which is a no-op when pending is empty: silent success would\n"
           "mislead callers who expect `flush()` to materialize a segment.\n"
           "Use `import_laser_segment` to publish precomputed LASER artifacts.")
      .def("import_laser_segment",
           &PyDiskCollection::import_laser_segment,
           py::arg("src_dir"),
           py::arg("labels"),
           py::kw_only(),
           py::arg("copy") = true,
           "Import a precomputed LASER native-artifact directory as a new segment.\n\n"
           "Requires the wrapped collection to have been constructed with\n"
           "`index_type=\"disk_laser\"` on a build where `disk_laser` is supported.\n\n"
           "Arguments:\n"
           "  src_dir: directory containing the four required LASER artifacts\n"
           "    (`*_R<R>_MD<main_dim>.index`, `_rotator`, `_cache_ids`,\n"
           "    `_cache_nodes`) plus optional sidecars (`_pca.bin`,\n"
           "    `_medoids`, `_medoids_indices`).\n"
           "  labels: numpy.uint64 1-D C-contiguous array of external labels\n"
           "    matching the LASER index's row count exactly.\n"
           "  copy (kw-only): v1 SHALL only accept `True`. `False` raises\n"
           "    `NotImplementedError`; hard-link import requires a future C++\n"
           "    API extension to accept `LaserSegmentImportParams`.\n\n"
           "Releases the GIL around the C++ import call. Raises:\n"
           "  RuntimeError: collection is not a `disk_laser` collection, or any\n"
           "    runtime error from the C++ importer (file open, copy, fsync).\n"
           "  ValueError: `src_dir` is not an existing directory, `labels` is\n"
           "    empty / wrong shape, or contains a duplicate label (within\n"
           "    batch or across previously-imported segments).\n"
           "  TypeError: `labels.dtype` is not `numpy.uint64`, or `labels`\n"
           "    is not C-contiguous.\n"
           "  NotImplementedError: `copy=False` (see above).")
      .def("search",
           &PyDiskCollection::search,
           py::arg("query"),
           py::arg("k") = 10,
           py::arg("ef") = 100,
           py::kw_only(),
           py::arg("beam_width") = 4,
           // The metric contract docstring is enforced by spec scenario
           // `test_disk_collection_cos_distance_docstring`; the literal phrases
           // below are required.
           "Return the top-k nearest neighbors as a list of (label, distance) tuples.\n\n"
           "Distance semantics (smaller is better):\n"
           "  L2: squared distance (Σ(qi - vi)^2)\n"
           "  IP: negative inner product (-Σ(qi * vi))\n"
           "  COS: negative cosine similarity after L2-normalizing stored vectors and query\n\n"
           "Argument k must be > 0; k > total_count returns total_count results.\n"
           "Argument ef must be > 0; Vamana uses ef as the greedy-search beam size.\n"
           "Argument beam_width (keyword-only, default 4) must be > 0; LASER uses\n"
           "  it as the libaio I/O parallelism setting. disk_flat / disk_vamana\n"
           "  ignore beam_width semantically (it is part of the engine-agnostic\n"
           "  DiskSearchOptions struct, but only LASER reads it).\n\n"
           "The query SHALL pass a finite-component check; a NaN or ±Inf value\n"
           "raises ValueError naming the offending position and value.")
      .def("batch_search",
           &PyDiskCollection::batch_search,
           py::arg("queries"),
           py::arg("k") = 10,
           py::arg("ef") = 100,
           py::kw_only(),
           py::arg("beam_width") = 4,
           py::arg("num_threads") = 0,
           "Run N queries in parallel and return the top-k labels as a (N, k) numpy.uint64 "
           "matrix.\n"
           "\n"
           "Inputs:\n"
           "  queries: numpy.ndarray, dtype=float32, shape=(N, dim), C-contiguous.\n"
           "  k: int > 0; top-k per query.\n"
           "  ef: int > 0; greedy-search beam size (engine-specific use, same as search()).\n"
           "  beam_width (kw-only, default 4): int > 0; LASER libaio I/O parallelism setting,\n"
           "    ignored by disk_flat / disk_vamana.\n"
           "  num_threads (kw-only, default 0): int >= 0. `0` auto-resolves to the\n"
           "    `OMP_NUM_THREADS` environment variable when it parses to an integer in\n"
           "    [1, 2**32 - 1], otherwise to `std::thread::hardware_concurrency()` with a\n"
           "    one-thread floor.\n"
           "\n"
           "Output:\n"
           "  labels: numpy.uint64 array of shape (N, k). For queries that yielded fewer\n"
           "    than k hits, trailing slots remain at the padding sentinel\n"
           "    `numpy.iinfo(numpy.uint64).max`.\n"
           "\n"
           "Validation: queries.dtype must be float32; queries.ndim must be 2;\n"
           "queries.shape[1] must equal dim(); queries must be C-contiguous; every element\n"
           "must be finite (NaN / +-Inf raises ValueError naming the offending row, col,\n"
           "and kind).\n"
           "\n"
           "Concurrency notes:\n"
           "  - Engine coverage is uniform across disk_flat, disk_vamana, and disk_laser.\n"
           "  - On disk_laser the per-segment search holds an internal mutex (see the\n"
           "    `disk-laser-searcher-thread-safety` contract), so multi-thread batch_search\n"
           "    does not scale throughput beyond the single-thread baseline; correctness\n"
           "    (labels match the per-query serial search) is preserved.\n"
           "  - The GIL is released around the C++ batch loop. The N=0 case returns an\n"
           "    empty (0, k) array; the C++ entry early-returns without spawning a worker\n"
           "    thread or reading the queries / output buffers.")
      .def("batch_search_with_distance",
           &PyDiskCollection::batch_search_with_distance,
           py::arg("queries"),
           py::arg("k") = 10,
           py::arg("ef") = 100,
           py::kw_only(),
           py::arg("beam_width") = 4,
           py::arg("num_threads") = 0,
           "Run N queries in parallel and return (labels, distances) as two (N, k) arrays.\n"
           "\n"
           "labels: numpy.uint64, shape (N, k); UINT64_MAX padding for short returns.\n"
           "distances: numpy.float32, shape (N, k); NaN padding for short returns.\n"
           "\n"
           "Distance contract:\n"
           "  - On disk_flat / disk_vamana every overwritten distance is the engine-native\n"
           "    distance value returned by the per-query search() (bit-exact against\n"
           "    `search()`'s second tuple element).\n"
           "  - On disk_laser every overwritten distance is NaN, consistent with the\n"
           "    `x_laser_distance_field_supported = false` contract; trailing slots remain\n"
           "    at the NaN sentinel for both engines.\n"
           "\n"
           "Validation, num_threads resolution (including `OMP_NUM_THREADS` lookup), GIL\n"
           "behavior, padding sentinels, and the `does not scale` note for disk_laser\n"
           "match `batch_search` exactly; only the return type differs.")
      .def("size", &PyDiskCollection::size)
      .def("dim", &PyDiskCollection::dim);
}

}  // namespace alaya::disk::pybindings
