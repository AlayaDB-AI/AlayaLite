#include "diskann.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "index/graph/diskann/diskann_index.hpp"

namespace alaya::diskann::pybindings {

namespace py = pybind11;

namespace {

void require_array(const py::array &array,
                   const py::dtype &dtype,
                   int dimensions,
                   const char *name) {
  if (!array.dtype().is(dtype)) {
    throw py::type_error(std::string(name) + " must have dtype " +
                         dtype.attr("name").cast<std::string>());
  }
  if (array.ndim() != dimensions) {
    throw py::value_error(std::string(name) + " must be " + std::to_string(dimensions) + "D");
  }
  if ((array.flags() & py::array::c_style) == 0) {
    throw py::value_error(std::string(name) + " must be C-contiguous");
  }
}

uint32_t checked_u32(py::ssize_t value, const char *name) {
  if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max()) {
    throw py::value_error(std::string(name) + " must fit uint32");
  }
  return static_cast<uint32_t>(value);
}

class PyDiskANNIndex {
 public:
  static void build(const std::string &path,
                    const py::array &vectors,
                    const py::array &labels,
                    const DiskANNBuildParams &params) {
    require_array(vectors, py::dtype::of<float>(), 2, "vectors");
    require_array(labels, py::dtype::of<uint64_t>(), 1, "labels");
    if (vectors.shape(0) == 0 || vectors.shape(1) == 0) {
      throw py::value_error("vectors must have non-zero rows and dimensions");
    }
    if (labels.shape(0) != vectors.shape(0)) {
      throw py::value_error("labels length must match vectors rows");
    }
    const auto count = static_cast<uint64_t>(vectors.shape(0));
    const auto dim = static_cast<uint64_t>(vectors.shape(1));
    const auto *vector_data = static_cast<const float *>(vectors.data());
    const auto *label_data = static_cast<const uint64_t *>(labels.data());
    py::gil_scoped_release release;
    DiskANNIndex::build(path, vector_data, label_data, count, dim, params);
  }

  static std::unique_ptr<PyDiskANNIndex> open(const std::string &path,
                                              const DiskANNLoadParams &params) {
    auto index = std::unique_ptr<PyDiskANNIndex>(new PyDiskANNIndex());
    {
      py::gil_scoped_release release;
      index->index_.load(path, params);
    }
    return index;
  }

  py::tuple search(const py::array &query,
                   uint32_t top_k,
                   const DiskANNSearchParams &params) const {
    require_array(query, py::dtype::of<float>(), 1, "query");
    if (query.shape(0) != static_cast<py::ssize_t>(index_.dim())) {
      throw py::value_error("query dimension does not match index dimension");
    }
    if (top_k == 0) {
      throw py::value_error("top_k must be > 0");
    }
    std::vector<uint64_t> labels(top_k);
    std::vector<float> distances(top_k);
    uint32_t count = 0;
    {
      py::gil_scoped_release release;
      count = index_.search(static_cast<const float *>(query.data()),
                            top_k,
                            labels.data(),
                            distances.data(),
                            params);
    }
    py::array_t<uint64_t> label_array(count);
    py::array_t<float> distance_array(count);
    std::memcpy(label_array.mutable_data(), labels.data(), count * sizeof(uint64_t));
    std::memcpy(distance_array.mutable_data(), distances.data(), count * sizeof(float));
    return py::make_tuple(std::move(label_array), std::move(distance_array));
  }

  py::tuple batch_search(const py::array &queries,
                         uint32_t top_k,
                         uint32_t num_threads,
                         const DiskANNSearchParams &params) const {
    require_array(queries, py::dtype::of<float>(), 2, "queries");
    if (queries.shape(0) == 0) {
      throw py::value_error("queries must have at least one row");
    }
    if (queries.shape(1) != static_cast<py::ssize_t>(index_.dim())) {
      throw py::value_error("query dimension does not match index dimension");
    }
    if (top_k == 0 || num_threads == 0) {
      throw py::value_error("top_k and num_threads must be > 0");
    }
    const uint32_t count = checked_u32(queries.shape(0), "queries rows");
    const size_t result_count = static_cast<size_t>(count) * top_k;
    std::vector<uint64_t> labels(result_count);
    std::vector<float> distances(result_count);
    {
      py::gil_scoped_release release;
      index_.batch_search(static_cast<const float *>(queries.data()),
                          count,
                          top_k,
                          labels.data(),
                          distances.data(),
                          num_threads,
                          params);
    }
    const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(count),
                                         static_cast<py::ssize_t>(top_k)};
    py::array_t<uint64_t> label_array(shape);
    py::array_t<float> distance_array(shape);
    std::memcpy(label_array.mutable_data(), labels.data(), result_count * sizeof(uint64_t));
    std::memcpy(distance_array.mutable_data(), distances.data(), result_count * sizeof(float));
    return py::make_tuple(std::move(label_array), std::move(distance_array));
  }

  void insert(const py::array &vector, uint64_t external_id) {
    require_vector(vector, "vector");
    py::gil_scoped_release release;
    index_.insert(static_cast<const float *>(vector.data()), external_id);
  }

  void batch_insert(const py::array &vectors, const py::array &external_ids, uint32_t batch_size) {
    require_array(vectors, py::dtype::of<float>(), 2, "vectors");
    require_array(external_ids, py::dtype::of<uint64_t>(), 1, "external_ids");
    if (vectors.shape(1) != static_cast<py::ssize_t>(index_.dim())) {
      throw py::value_error("vectors dimension does not match index dimension");
    }
    if (vectors.shape(0) != external_ids.shape(0)) {
      throw py::value_error("external_ids length must match vectors rows");
    }
    const uint32_t count = checked_u32(vectors.shape(0), "vectors rows");
    py::gil_scoped_release release;
    index_.batch_insert(static_cast<const float *>(vectors.data()),
                        static_cast<const uint64_t *>(external_ids.data()),
                        count,
                        batch_size);
  }

  void remove(uint64_t external_id) {
    try {
      py::gil_scoped_release release;
      index_.remove_by_label(external_id);
    } catch (const std::out_of_range &error) {
      throw py::key_error(error.what());
    }
  }

  void batch_remove(const py::array &external_ids) {
    require_array(external_ids, py::dtype::of<uint64_t>(), 1, "external_ids");
    const uint32_t count = checked_u32(external_ids.shape(0), "external_ids length");
    try {
      py::gil_scoped_release release;
      index_.batch_remove_by_labels(static_cast<const uint64_t *>(external_ids.data()), count);
    } catch (const std::out_of_range &error) {
      throw py::key_error(error.what());
    }
  }

  bool contains(uint64_t external_id) const { return index_.contains_label(external_id); }

  void flush() {
    py::gil_scoped_release release;
    index_.flush();
  }

  uint64_t size() const { return index_.size(); }
  uint64_t dim() const { return index_.dim(); }
  bool updatable() const { return index_.updatable(); }

 private:
  void require_vector(const py::array &vector, const char *name) const {
    require_array(vector, py::dtype::of<float>(), 1, name);
    if (vector.shape(0) != static_cast<py::ssize_t>(index_.dim())) {
      throw py::value_error(std::string(name) + " dimension does not match index dimension");
    }
  }

  DiskANNIndex index_;
};

}  // namespace

void register_diskann(py::module_ &module) {
  py::enum_<DiskANNUpdateIO>(module, "UpdateIO")
      .value("AUTO", DiskANNUpdateIO::kAuto)
      .value("URING", DiskANNUpdateIO::kUring)
      .value("BLOCKING", DiskANNUpdateIO::kBlocking);

  py::class_<DiskANNBuildParams>(module, "BuildParams")
      .def(py::init<>())
      .def_readwrite("R", &DiskANNBuildParams::R)
      .def_readwrite("L", &DiskANNBuildParams::L)
      .def_readwrite("alpha", &DiskANNBuildParams::alpha)
      .def_readwrite("record_capacity", &DiskANNBuildParams::record_capacity)
      .def_readwrite("pq_n_chunks", &DiskANNBuildParams::pq_n_chunks)
      .def_readwrite("cache_ratio", &DiskANNBuildParams::cache_ratio)
      .def_readwrite("num_threads", &DiskANNBuildParams::num_threads)
      .def_readwrite("pq_train_iters", &DiskANNBuildParams::pq_train_iters)
      .def_readwrite("seed", &DiskANNBuildParams::seed)
      .def_readwrite("verbose", &DiskANNBuildParams::verbose);

  py::class_<DiskANNLoadParams>(module, "LoadParams")
      .def(py::init<>())
      .def_readwrite("num_threads", &DiskANNLoadParams::num_threads)
      .def_readwrite("beam_width", &DiskANNLoadParams::beam_width)
      .def_readwrite("nopq_io_depth", &DiskANNLoadParams::nopq_io_depth)
      .def_readwrite("scratch_search_list_size", &DiskANNLoadParams::scratch_search_list_size)
      .def_readwrite("updatable", &DiskANNLoadParams::updatable)
      .def_readwrite("update_search_l", &DiskANNLoadParams::update_search_l)
      .def_readwrite("update_rerank", &DiskANNLoadParams::update_rerank)
      .def_readwrite("update_insert_prune", &DiskANNLoadParams::update_insert_prune)
      .def_readwrite("update_alpha", &DiskANNLoadParams::update_alpha)
      .def_readwrite("safety_net_ratio", &DiskANNLoadParams::safety_net_ratio)
      .def_readwrite("safety_net_ops", &DiskANNLoadParams::safety_net_ops)
      .def_readwrite("page_cache_capacity", &DiskANNLoadParams::page_cache_capacity)
      .def_readwrite("update_insert_threads", &DiskANNLoadParams::update_insert_threads)
      .def_readwrite("update_reconnect_threads", &DiskANNLoadParams::update_reconnect_threads)
      .def_readwrite("update_io", &DiskANNLoadParams::update_io)
      .def_readwrite("update_search_concurrency", &DiskANNLoadParams::update_search_concurrency)
      .def_readwrite("search_page_cache", &DiskANNLoadParams::search_page_cache);

  py::class_<DiskANNSearchParams>(module, "SearchParams")
      .def(py::init<>())
      .def_readwrite("search_list_size", &DiskANNSearchParams::search_list_size)
      .def_readwrite("use_pq", &DiskANNSearchParams::use_pq)
      .def_readwrite("rerank", &DiskANNSearchParams::rerank)
      .def_readwrite("rerank_count", &DiskANNSearchParams::rerank_count)
      .def_readwrite("deterministic", &DiskANNSearchParams::deterministic);

  DiskANNLoadParams default_load_params;
  default_load_params.updatable = true;
  py::class_<PyDiskANNIndex>(module, "Index")
      .def_static("build",
                  &PyDiskANNIndex::build,
                  py::arg("path"),
                  py::arg("vectors"),
                  py::arg("external_ids"),
                  py::arg("params") = DiskANNBuildParams{})
      .def_static("open",
                  &PyDiskANNIndex::open,
                  py::arg("path"),
                  py::arg("params") = default_load_params)
      .def("search",
           &PyDiskANNIndex::search,
           py::arg("query"),
           py::arg("top_k") = 10,
           py::arg("params") = DiskANNSearchParams{})
      .def("batch_search",
           &PyDiskANNIndex::batch_search,
           py::arg("queries"),
           py::arg("top_k") = 10,
           py::kw_only(),
           py::arg("num_threads") = 1,
           py::arg("params") = DiskANNSearchParams{})
      .def("insert", &PyDiskANNIndex::insert, py::arg("vector"), py::arg("external_id"))
      .def("batch_insert",
           &PyDiskANNIndex::batch_insert,
           py::arg("vectors"),
           py::arg("external_ids"),
           py::kw_only(),
           py::arg("batch_size") = 32)
      .def("remove", &PyDiskANNIndex::remove, py::arg("external_id"))
      .def("batch_remove", &PyDiskANNIndex::batch_remove, py::arg("external_ids"))
      .def("contains", &PyDiskANNIndex::contains, py::arg("external_id"))
      .def("flush", &PyDiskANNIndex::flush)
      .def_property_readonly("size", &PyDiskANNIndex::size)
      .def_property_readonly("dim", &PyDiskANNIndex::dim)
      .def_property_readonly("updatable", &PyDiskANNIndex::updatable);
}

}  // namespace alaya::diskann::pybindings
