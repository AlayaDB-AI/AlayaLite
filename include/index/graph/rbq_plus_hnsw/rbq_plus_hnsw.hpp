#pragma once

#include <immintrin.h>
#include <omp.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#include "buffer.hpp"
#include "space/rbq_space/rbq_space.hpp"
#include "utils/metric_type.hpp"
#include "utils/prefetch.hpp"
#include "utils/rbq_utils/parallel.hpp"
#include "visited_pool.hpp"

// NOLINTBEGIN
namespace alaya {
template <typename T>
using maxheap = std::priority_queue<T>;

template <typename T>
using minheap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

// Bounded priority queue implemented as a sorted vector.

template <typename DataType = float, typename IDType = PID, typename DistanceType = float>
class RBQ_HNSW {  // NOLINT
 private:
  // NOLINTBEGIN
  struct ResultRecord {
    DistanceType est_dist;
    DistanceType low_dist;
    ResultRecord(DistanceType est_dist, DistanceType low_dist)
        : est_dist(est_dist), low_dist(low_dist) {}
    auto operator<(const ResultRecord &other) const -> bool {
      return this->est_dist < other.est_dist;
    }
  };

  struct Candidate {
    ResultRecord record;
    IDType id;
  };

  class BoundedKNN {
   public:
    explicit BoundedKNN(size_t capacity) : capacity_(capacity) {}

    // Insert a candidate in sorted order (ascending by est_dist).
    void insert(const Candidate &cand) {
      // Find insertion position using binary search.
      auto it = std::upper_bound(queue_.begin(), queue_.end(), cand,
                                 [](const Candidate &a, const Candidate &b) {
                                   return a.record.est_dist < b.record.est_dist;
                                 });
      queue_.insert(it, cand);
      // If we exceed capacity, drop the worst candidate (largest est_dist).
      if (queue_.size() > capacity_) {
        queue_.pop_back();
      }
    }

    // Returns the worst (largest est_dist) candidate.
    [[nodiscard]] auto worst() const -> const Candidate & { return queue_.back(); }

    [[nodiscard]] auto size() const -> size_t { return queue_.size(); }

    [[nodiscard]] const std::vector<Candidate> &candidates() const { return queue_; }

   private:
    size_t capacity_;
    // Sorted in ascending order by record.est_dist so that the worst is at the back.
    std::vector<Candidate> queue_;
  };

  static constexpr IDType kMaxLabelOperationLock = 65536;
  size_t max_elements_{0};
  mutable std::atomic<size_t> cur_element_count_{0};  // current number of elements
  size_t size_data_per_element_{0};
  size_t size_links_per_element_{0};
  size_t M_{0};
  size_t maxM_{0};
  size_t maxM0_{0};
  size_t ef_construction_{0};
  size_t ef_{0};

  double mult_{0.0}, revSize_{0.0};
  int maxlevel_{0};
  // Locks operations with element by label value
  mutable std::vector<std::mutex> label_op_locks_;

  std::mutex global_;
  std::vector<std::mutex> link_list_locks_;

  IDType enterpoint_node_{0};

  size_t size_links_level0_{0};

  size_t offsetBinData_{0}, offsetExData_{0}, label_offset_{0};
  size_t size_bin_data_{0}, size_ex_data_{0};
  size_t ex_bits_{0};

  // Layout: (# of edges + edges) + (cluster_id) + (External_id) + (BinData) + (ExData)
  char *data_level0_memory_{nullptr};
  char **linkLists_{nullptr};
  std::vector<int> element_levels_;  // keeps level of each element

  size_t num_cluster_{0};
  size_t dim_{0};
  size_t padded_dim_{0};

  char *centroids_memory_{nullptr};

  mutable std::mutex label_lookup_lock_;  // lock for label_lookup_
  std::unordered_map<IDType, IDType> label_lookup_;

  std::default_random_engine level_generator_;
  std::default_random_engine update_probability_generator_;

  std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};
  rabitq_impl::RabitqConfig query_config_;

  std::unique_ptr<RBQSpace<DataType, IDType, DistanceType>> space_ = nullptr;

  const DataType *rawDataPtr_{nullptr};

  bool is_valid = false;

  void free_memory() {
    if (!is_valid) {
      return;
    }
    free(data_level0_memory_);
    data_level0_memory_ = nullptr;
    for (IDType i = 0; i < cur_element_count_; i++) {
      if (element_levels_[i] > 0) {
        free(linkLists_[i]);
      }
    }
    free(reinterpret_cast<void *>(linkLists_));
    linkLists_ = nullptr;
    cur_element_count_ = 0;

    free(centroids_memory_);

    space_.reset();

    is_valid = false;
  }

  void set_ef(size_t ef) { ef_ = ef; }

  auto get_label_op_mutex(IDType label) const -> std::mutex & {
    // calculate hash
    size_t lock_id = label & (kMaxLabelOperationLock - 1);
    return label_op_locks_[lock_id];
  }

  auto get_external_label(IDType internal_id) const -> IDType {
    IDType return_label;
    memcpy(&return_label,
           (data_level0_memory_ + (internal_id * size_data_per_element_) + label_offset_),
           sizeof(IDType));
    return return_label;
  }

  void set_external_label(IDType internal_id, IDType label) const {
    memcpy((data_level0_memory_ + (internal_id * size_data_per_element_) + label_offset_), &label,
           sizeof(IDType));
  }

  auto get_external_label_pt(IDType internal_id) const -> IDType * {
    return reinterpret_cast<IDType *>(data_level0_memory_ + (internal_id * size_data_per_element_) +
                                      label_offset_);
  }

  auto get_bindata_by_internalid(IDType internal_id) const -> char * {
    return reinterpret_cast<char *>(data_level0_memory_ + (internal_id * size_data_per_element_) +
                                    offsetBinData_);
  }

  auto get_exdata_by_internalid(IDType internal_id) const -> char * {
    return reinterpret_cast<char *>(data_level0_memory_ + (internal_id * size_data_per_element_) +
                                    offsetExData_);
  }

  auto get_clusterid_by_internalid(IDType internal_id) const -> IDType {
    return *(reinterpret_cast<IDType *>(
        data_level0_memory_ + (internal_id * size_data_per_element_) + size_links_level0_));
  }

  auto get_clusterid_pt(IDType internal_id) const -> char * {
    return reinterpret_cast<char *>(data_level0_memory_ + (internal_id * size_data_per_element_) +
                                    size_links_level0_);
  }

  auto get_random_level(double reverse_size) -> int {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return static_cast<int>(r);
  }

  auto get_max_elements() const -> size_t { return max_elements_; }

  auto get_current_element_count() const -> size_t { return cur_element_count_; }

  auto get_linklist(IDType internal_id, int level) const -> IDType * {
    return reinterpret_cast<IDType *>(linkLists_[internal_id] +
                                      ((level - 1) * size_links_per_element_));
  }

  auto get_linklist0(IDType internal_id) const -> IDType * {
    return reinterpret_cast<IDType *>(data_level0_memory_ + (internal_id * size_data_per_element_));
  }

  static auto get_list_count(const IDType *ptr) -> unsigned short int {
    return *(reinterpret_cast<const unsigned short int *>(ptr));
  }

  static void set_list_count(IDType *ptr, unsigned short int size) {
    *(reinterpret_cast<unsigned short int *>(ptr)) = size;
  }

  DistanceType get_data_dist(IDType obj1, IDType obj2) {
    IDType label1 = get_external_label(obj1);
    IDType label2 = get_external_label(obj2);
    auto dist_func = space_->get_dist_func();
    return dist_func(rawDataPtr_ + (label1 * dim_), rawDataPtr_ + (label2 * dim_), dim_);
  }

  void add_point(IDType label, IDType cluster_id, const rabitq_impl::RabitqConfig &config) {
    std::unique_lock<std::mutex> lock_label(get_label_op_mutex(label));

    int level = -1;
    IDType cur_c = 0;
    {
      std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
      if (label_lookup_.find(label) != label_lookup_.end()) {
        throw std::runtime_error(
            "Currently not support replacement of existing elements, only support "
            "inserting elements with distinct labels");
      }

      if (cur_element_count_ >= max_elements_) {
        throw std::runtime_error("The number of elements exceeds the specified limit");
      }

      cur_c = cur_element_count_;
      cur_element_count_++;
      label_lookup_[label] = cur_c;
    }

    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
    int curlevel = get_random_level(mult_);
    if (level > 0) {
      curlevel = level;
    }

    element_levels_[cur_c] = curlevel;
    std::unique_lock<std::mutex> templock(global_);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy) {
      templock.unlock();
    }
    IDType curr_obj = enterpoint_node_;

    // initialize the current memory.
    memset(data_level0_memory_ + (cur_c * size_data_per_element_), 0, size_data_per_element_);

    // Initialisation of label and cluster id
    memcpy(get_external_label_pt(cur_c), &label, sizeof(IDType));
    memcpy(get_clusterid_pt(cur_c), &cluster_id, sizeof(IDType));

    // Quantize raw data and initialize quantized data(space's task)
    std::vector<DataType> rotated_data(padded_dim_);
    space_->rotate_data(rawDataPtr_ + (label * dim_), rotated_data.data());
    space_->quantize_split_single(
        rotated_data.data(),
        reinterpret_cast<DataType *>(centroids_memory_) + (cluster_id * padded_dim_), ex_bits_,
        get_bindata_by_internalid(cur_c), get_exdata_by_internalid(cur_c), config);

    // If the current vertex is at level >0, it needs some space to store the extra edges.
    if (curlevel > 0) {
      linkLists_[cur_c] = static_cast<char *>(malloc((size_links_per_element_ * curlevel) + 1));
      if (linkLists_[cur_c] == nullptr) {
        throw std::runtime_error("Not enough memory: add_point failed to allocate linklist");
      }
      memset(linkLists_[cur_c], 0, (size_links_per_element_ * curlevel) + 1);
    }

    if (static_cast<signed>(curr_obj) != -1) {
      if (curlevel < maxlevelcopy) {
        DistanceType curdist = get_data_dist(curr_obj, cur_c);
        for (int level = maxlevelcopy; level > curlevel; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int *data;
            std::unique_lock<std::mutex> lock(link_list_locks_[curr_obj]);
            data = get_linklist(curr_obj, level);
            int size = get_list_count(data);

            auto *datal = static_cast<IDType *>(data + 1);
            for (int i = 0; i < size; i++) {
              IDType cand = datal[i];
              if (cand > max_elements_) {
                throw std::runtime_error("cand error");
              }
              DistanceType d = get_data_dist(cand, cur_c);
              if (d < curdist) {
                curdist = d;
                curr_obj = cand;
                changed = true;
              }
            }
          }
        }
      }

      for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
        maxheap<std::pair<DistanceType, IDType>> top_candidates =
            search_base_layer(curr_obj, cur_c, level);
        curr_obj = mutually_connect_new_element(cur_c, top_candidates, level);
      }
    } else {
      // Do nothing for the first element
      enterpoint_node_ = 0;
      maxlevel_ = curlevel;
    }

    // Releasing lock for the maximum level
    if (curlevel > maxlevelcopy) {
      enterpoint_node_ = cur_c;
      maxlevel_ = curlevel;
    }
  }

  maxheap<std::pair<DistanceType, IDType>> search_base_layer(IDType ep_id, IDType cur_c,
                                                             int layer) {
    HashBasedBooleanSet *vl = visited_list_pool_->get_free_vislist();

    maxheap<std::pair<DistanceType, IDType>> top_candidates;
    minheap<std::pair<DistanceType, IDType>> candidate_set;

    DistanceType lower_bound = get_data_dist(ep_id, cur_c);
    top_candidates.emplace(lower_bound, ep_id);
    candidate_set.emplace(lower_bound, ep_id);
    vl->set(ep_id);

    while (!candidate_set.empty()) {
      std::pair<DistanceType, IDType> curr_el_pair = candidate_set.top();
      if (curr_el_pair.first > lower_bound && top_candidates.size() == ef_construction_) {
        break;
      }
      candidate_set.pop();

      IDType cur_node_num = curr_el_pair.second;

      std::unique_lock<std::mutex> lock(link_list_locks_[cur_node_num]);

      int *data;
      if (layer == 0) {
        data = reinterpret_cast<int *>(get_linklist0(cur_node_num));
      } else {
        data = reinterpret_cast<int *>(get_linklist(cur_node_num, layer));
      }
      size_t size = get_list_count(reinterpret_cast<IDType *>(data));
      auto *datal = reinterpret_cast<IDType *>(data + 1);

      mem_prefetch_l1(reinterpret_cast<const char *>(rawDataPtr_ + (get_external_label(*datal) * dim_)),
                      padded_dim_ / 16);

      mem_prefetch_l1(
          reinterpret_cast<const char *>(rawDataPtr_ + (get_external_label(*(datal + 1)) * dim_)),
          padded_dim_ / 16);

      for (size_t j = 0; j < size; j++) {
        IDType candidate_id = *(datal + j);
        if (vl->get(candidate_id)) {
          continue;
        }
        vl->set(candidate_id);

        if (j < size - 1) {
          mem_prefetch_l1(
              reinterpret_cast<const char *>(rawDataPtr_ + (get_external_label(*(datal + j + 1)) * dim_)),
              padded_dim_ / 16);
        }

        DistanceType dist1 = get_data_dist(candidate_id, cur_c);
        if (top_candidates.size() < ef_construction_ || lower_bound > dist1) {
          candidate_set.emplace(dist1, candidate_id);
          top_candidates.emplace(dist1, candidate_id);
          if (top_candidates.size() > ef_construction_) {
            top_candidates.pop();
          }
          if (!top_candidates.empty()) {
            lower_bound = top_candidates.top().first;
          }
        }
      }
    }
    visited_list_pool_->release_vis_list(vl);
    return top_candidates;
  }

  IDType mutually_connect_new_element(IDType cur_c,
                                      maxheap<std::pair<DistanceType, IDType>> &top_candidates,
                                      int level) {
    size_t max_m = level > 0 ? maxM_ : maxM0_;
    get_neighbors_by_heuristic2(top_candidates, M_);
    if (top_candidates.size() > M_) {
      throw std::runtime_error(
          "Should be not be more than M_ candidates returned by the heuristic");
    }

    std::vector<IDType> selected_neighbors;
    selected_neighbors.reserve(M_);
    while (top_candidates.size() > 0) {
      selected_neighbors.push_back(top_candidates.top().second);
      top_candidates.pop();
    }

    IDType next_closest_entry_point = selected_neighbors.back();

    {
      IDType *ll_cur;
      if (level == 0) {
        ll_cur = get_linklist0(cur_c);
      } else {
        ll_cur = get_linklist(cur_c, level);
      }

      if (*ll_cur > 0) {
        throw std::runtime_error("The newly inserted element should have blank link list");
      }

      set_list_count(ll_cur, selected_neighbors.size());
      auto *data = static_cast<IDType *>(ll_cur + 1);
      for (size_t idx = 0; idx < selected_neighbors.size(); idx++) {
        if (data[idx] != 0) {
          throw std::runtime_error("Possible memory corruption");
        }
        if (level > element_levels_[selected_neighbors[idx]]) {
          throw std::runtime_error("Trying to make a link on a non-existent level");
        }

        data[idx] = selected_neighbors[idx];
      }
    }

    for (auto selected_neighbor : selected_neighbors) {
      std::unique_lock<std::mutex> lock(link_list_locks_[selected_neighbor]);

      IDType *ll_other;
      if (level == 0) {
        ll_other = get_linklist0(selected_neighbor);
      } else {
        ll_other = get_linklist(selected_neighbor, level);
      }

      size_t sz_link_list_other = get_list_count(ll_other);

      if (sz_link_list_other > max_m) {
        throw std::runtime_error("Bad value of sz_link_list_other");
      }
      if (selected_neighbor == cur_c) {
        throw std::runtime_error("Trying to connect an element to itself");
      }
      if (level > element_levels_[selected_neighbor]) {
        throw std::runtime_error("Trying to make a link on a non-existent level");
      }

      auto *data = static_cast<IDType *>(ll_other + 1);

      bool is_cur_c_present = false;
      for (size_t j = 0; j < sz_link_list_other; j++) {
        if (data[j] == cur_c) {
          is_cur_c_present = true;
          break;
        }
      }

      if (!is_cur_c_present) {
        if (sz_link_list_other < max_m) {
          data[sz_link_list_other] = cur_c;
          set_list_count(ll_other, sz_link_list_other + 1);
        } else {
          DistanceType d_max = get_data_dist(selected_neighbor, cur_c);
          maxheap<std::pair<DistanceType, IDType>> candidates;
          candidates.emplace(d_max, cur_c);
          for (size_t j = 0; j < sz_link_list_other; j++) {
            candidates.emplace(get_data_dist(data[j], selected_neighbor), data[j]);
          }

          get_neighbors_by_heuristic2(candidates, max_m);

          int indx = 0;
          while (candidates.size() > 0) {
            data[indx] = candidates.top().second;
            candidates.pop();
            indx++;
          }

          set_list_count(ll_other, indx);
        }
      }
    }

    return next_closest_entry_point;
  }

  void get_neighbors_by_heuristic2(maxheap<std::pair<DistanceType, IDType>> &top_candidates,
                                   size_t M) {
    if (top_candidates.size() < M) {
      return;
    }

    minheap<std::pair<DistanceType, IDType>> queue_closest;
    std::vector<std::pair<DistanceType, IDType>> return_list;
    while (top_candidates.size() > 0) {
      queue_closest.emplace(top_candidates.top());
      top_candidates.pop();
    }

    while (queue_closest.size() > 0) {
      if (return_list.size() >= M) {
        break;
      }
      std::pair<DistanceType, IDType> current_pair = queue_closest.top();
      DistanceType dist_to_query = current_pair.first;
      queue_closest.pop();
      bool good = true;

      for (std::pair<DistanceType, IDType> second_pair : return_list) {
        DistanceType curdist = get_data_dist(second_pair.second, current_pair.second);
        if (curdist < dist_to_query) {
          good = false;
          break;
        }
      }
      if (good) {
        return_list.push_back(current_pair);
      }
    }

    for (std::pair<DistanceType, IDType> current_pair : return_list) {
      top_candidates.emplace(current_pair);
    }
  }

  // Optimized search function.
  void searchBaseLayerST_AdaptiveRerankOpt(IDType ep_id, size_t ef, size_t TOPK,
                                           SplitSingleQuery<DataType> &query_wrapper,
                                           std::vector<DistanceType> &q_to_centroids,  // preprocess
                                           [[maybe_unused]] const DataType *query,
                                           BoundedKNN &boundedKNN) {
    HashBasedBooleanSet *vl = visited_list_pool_->get_free_vislist();

    // Use bounded priority queue instead of the maxheap.
    SearchBuffer<> candidate_set(ef);

    float distk = 1e10;

    EstimateRecord start_estimate_record;
    space_->get_full_est(q_to_centroids, query_wrapper, start_estimate_record, num_cluster_,
                         ex_bits_, get_clusterid_by_internalid(ep_id),
                         get_bindata_by_internalid(ep_id), get_exdata_by_internalid(ep_id));
    DistanceType est_dist = start_estimate_record.est_dist;
    DistanceType low_dist = start_estimate_record.low_dist;

    // Insert initial candidate.
    boundedKNN.insert({ResultRecord(est_dist, low_dist), ep_id});
    candidate_set.insert(ep_id, est_dist);

    distk = est_dist;

    vl->set(ep_id);

    while (candidate_set.has_next()) {
      // Step 1 - get the next node to explore.
      IDType current_node_id = candidate_set.pop();
      int *data = (int *)get_linklist0(current_node_id);
      size_t size = get_list_count((IDType *)data);

      mem_prefetch_l1(get_bindata_by_internalid(*(data + 1)), 2);
      // Iterate over neighbors. (List starts at index 1.)
      for (size_t j = 1; j <= size; j++) {
        int candidate_id = *(data + j);

        mem_prefetch_l1(get_bindata_by_internalid(*(data + j + 1)), 2);

        if (!vl->get(candidate_id)) {
          vl->set(candidate_id);
          EstimateRecord candest;
          space_->get_bin_est(q_to_centroids, query_wrapper, candest, num_cluster_,
                              get_clusterid_by_internalid(candidate_id),
                              get_bindata_by_internalid(candidate_id));

          if (ex_bits_ > 0) {
            // Check preliminary score against current worst full estimate.
            bool flag_update_KNNs = boundedKNN.size() < TOPK || candest.low_dist < distk;

            if (flag_update_KNNs) {
              // Compute the full estimate if promising.
              space_->get_full_est(q_to_centroids, query_wrapper, candest, num_cluster_, ex_bits_,
                                   get_clusterid_by_internalid(candidate_id),
                                   get_bindata_by_internalid(candidate_id),
                                   get_exdata_by_internalid(candidate_id));
              Candidate cand{ResultRecord(candest.est_dist, candest.low_dist),
                             static_cast<IDType>(candidate_id)};
              boundedKNN.insert(cand);
              distk = boundedKNN.worst().record.est_dist;
            }
          } else {
            Candidate cand{ResultRecord(candest.est_dist, candest.low_dist),
                           static_cast<IDType>(candidate_id)};
            boundedKNN.insert(cand);
          }

          if (!candidate_set.is_full(candest.est_dist)) {
            candidate_set.insert(candidate_id, candest.est_dist);
          }

          mem_prefetch_l2((char *)get_linklist0(candidate_set.next_id()), 2);
        }
      }
    }

    visited_list_pool_->release_vis_list(vl);
  }

  maxheap<std::pair<DistanceType, IDType>> search_knn(const DataType *rotated_query, size_t TOPK) {
    maxheap<std::pair<DistanceType, IDType>> result;
    if (cur_element_count_ == 0) {
      return result;
    }

    MetricType metric_type = space_->get_metric_type();
    SplitSingleQuery<DataType> query_wrapper(rotated_query, padded_dim_, ex_bits_, query_config_,
                                             metric_type);

    // Preprocess - get the distance from query to all centroids
    std::vector<DistanceType> q_to_centroids(num_cluster_);

    if (metric_type == MetricType::L2) {
      for (size_t i = 0; i < num_cluster_; i++) {
        q_to_centroids[i] = std::sqrt(space_->get_distance(
            rotated_query, reinterpret_cast<DataType *>(centroids_memory_) + (i * padded_dim_),
            padded_dim_));
      }
    } else if (metric_type == MetricType::IP || metric_type == MetricType::COS) {
      q_to_centroids.resize(2 * num_cluster_);
      // first half as g_add, second half as g_error
      for (size_t i = 0; i < num_cluster_; i++) {
        q_to_centroids[i] = dot_product(
            rotated_query, reinterpret_cast<DataType *>(centroids_memory_) + (i * padded_dim_),
            padded_dim_);
        q_to_centroids[i + num_cluster_] = std::sqrt(euclidean_sqr(
            rotated_query, reinterpret_cast<DataType *>(centroids_memory_) + (i * padded_dim_),
            padded_dim_));
      }
    }

    IDType curr_obj = enterpoint_node_;
    EstimateRecord curest;

    space_->get_bin_est(q_to_centroids, query_wrapper, curest, num_cluster_,
                        get_clusterid_by_internalid(curr_obj), get_bindata_by_internalid(curr_obj));

    for (int level = maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        unsigned int *data;

        data = static_cast<unsigned int *>(get_linklist(curr_obj, level));
        int size = get_list_count(data);

        IDType *datal = static_cast<IDType *>(data + 1);
        for (int i = 0; i < size; i++) {
          IDType cand = datal[i];
          if (cand > max_elements_) {
            throw std::runtime_error("cand error");
          }

          EstimateRecord candest;
          space_->get_bin_est(q_to_centroids, query_wrapper, candest, num_cluster_,
                              get_clusterid_by_internalid(cand), get_bindata_by_internalid(cand));

          if (candest.est_dist < curest.est_dist) {
            curest = candest;
            curr_obj = cand;
            changed = true;
          }
        }
      }
    }

    BoundedKNN boundedKnn(TOPK);
    searchBaseLayerST_AdaptiveRerankOpt(curr_obj, std::max(ef_, TOPK), TOPK, query_wrapper,
                                        q_to_centroids, rotated_query, boundedKnn);
    for (auto &candidate : boundedKnn.candidates()) {
      result.emplace(candidate.record.est_dist, get_external_label(candidate.id));
    }
    return result;
  }

 public:
  RBQ_HNSW() = default;
  ~RBQ_HNSW() { free_memory(); }

  RBQ_HNSW(size_t max_elements, size_t dim, size_t total_bits, size_t M, size_t ef_construction,
           size_t random_seed, MetricType mt, RotatorType rt)
      : label_op_locks_(kMaxLabelOperationLock),
        link_list_locks_(max_elements),
        element_levels_(max_elements),
        dim_(dim),
        max_elements_(max_elements),
        ex_bits_(total_bits - 1) {
    if (total_bits < 1 || total_bits > 9) {
      std::cerr << "Invalid number of bits for quantization in "
                   "HierarchicalNSW::HierarchicalNSW\n";
      std::cerr << "Expected: 1 to 9  Input:" << total_bits << '\n';
      std::cerr.flush();
      exit(1);
    };

    space_ = std::make_unique<RBQSpace<DataType, IDType, DistanceType>>(dim, mt, rt);
    padded_dim_ = space_->get_rotator_size();
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dim_);

    if (M <= 10000) {
      M_ = M;
    } else {
      std::cout << "warning: M parameter exceeds 10000 which may lead to adverse effects." << '\n';
      std::cout << "Cap to 10000 will be applied for the rest of the processing." << '\n';
      M_ = 10000;
    }
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    size_bin_data_ = BinDataMap<DataType>::data_bytes(padded_dim_);
    size_ex_data_ = ExDataMap<DataType>::data_bytes(padded_dim_, ex_bits_);
    size_links_level0_ = maxM0_ * sizeof(IDType) + sizeof(IDType);
    label_offset_ = size_links_level0_ + sizeof(IDType);  // (# of edges + edges) + (cluster_id)
    offsetBinData_ =
        label_offset_ + sizeof(IDType);  // (# of edges + edges) + (cluster_id) + (external label)
    offsetExData_ = offsetBinData_ + size_bin_data_;  // (# of edges + edges) + (cluster_id)
                                                      // + (external label) + (BinData)
    size_data_per_element_ =
        offsetExData_ + size_ex_data_;  // (# of edges + edges) + (cluster_id) + (external
                                        // label) + (BinData) + (ExData)
    data_level0_memory_ = reinterpret_cast<char *>(malloc(max_elements_ * size_data_per_element_));
    if (data_level0_memory_ == nullptr) {
      throw std::runtime_error("Not enough memory");
    }

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    cur_element_count_ = 0;

    visited_list_pool_ = std::make_unique<VisitedListPool>(1, max_elements_);

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = reinterpret_cast<char **>(malloc(sizeof(void *) * max_elements_));
    if (linkLists_ == nullptr) {
      throw std::runtime_error("Not enough memory: HNSW failed to allocate linklists");
    }
    size_links_per_element_ = maxM_ * sizeof(IDType) + sizeof(IDType);
    mult_ = 1 / log(1.0 * static_cast<double>(M_));
    revSize_ = 1.0 / mult_;

    this->query_config_ =
        alaya::rabitq_impl::faster_config(padded_dim_, SplitSingleQuery<DataType>::kNumBits);

    is_valid = true;
  }

  void save(const char *filename) const {
    std::ofstream output(filename, std::ios::binary);
    space_->save(output);

    output.write(reinterpret_cast<const char *>(&max_elements_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&cur_element_count_), sizeof(size_t));

    output.write(reinterpret_cast<const char *>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&padded_dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&num_cluster_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&ex_bits_), sizeof(size_t));

    output.write(reinterpret_cast<const char *>(&size_bin_data_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&size_ex_data_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&size_links_level0_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&offsetBinData_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&offsetExData_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&label_offset_),
                 sizeof(size_t));  // possible error: (original->IDType)
    output.write(reinterpret_cast<const char *>(&size_data_per_element_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&size_links_per_element_), sizeof(size_t));

    output.write(reinterpret_cast<const char *>(&maxlevel_), sizeof(int));
    output.write(reinterpret_cast<const char *>(&enterpoint_node_), sizeof(IDType));

    output.write(reinterpret_cast<const char *>(&M_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&maxM_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&maxM0_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&mult_), sizeof(double));
    output.write(reinterpret_cast<const char *>(&ef_construction_), sizeof(size_t));

    std::cout << "cur_element_count = " << cur_element_count_ << '\n';

    output.write(reinterpret_cast<const char *>(centroids_memory_),
                 num_cluster_ * padded_dim_ * sizeof(DataType));

    output.write(reinterpret_cast<const char *>(data_level0_memory_),
                 cur_element_count_ * size_data_per_element_);

    for (size_t i = 0; i < cur_element_count_; i++) {
      unsigned int link_list_size =
          element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
      output.write(reinterpret_cast<const char *>(&link_list_size), sizeof(unsigned int));
      if (link_list_size != 0) {
        output.write(reinterpret_cast<const char *>(linkLists_[i]), link_list_size);
      }
    }

    output.close();
  }

  void load(const char *filename) {
    std::ifstream input(filename, std::ios::binary);

    if (!input.is_open()) {
      throw std::runtime_error("Cannot open file");
    }

    free_memory();

    space_ = std::make_unique<RBQSpace<DataType, IDType, DistanceType>>();
    space_->load(input);

    input.read(reinterpret_cast<char *>(&max_elements_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&cur_element_count_), sizeof(size_t));

    input.read(reinterpret_cast<char *>(&dim_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&padded_dim_), sizeof(size_t));
    if (space_->get_rotator_size() != padded_dim_) {
      std::cerr << "Bad padded_dim_ for rotator in hnsw.load()\n";
      exit(1);
    }
    input.read(reinterpret_cast<char *>(&num_cluster_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&ex_bits_), sizeof(size_t));

    input.read(reinterpret_cast<char *>(&size_bin_data_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&size_ex_data_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&size_links_level0_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&offsetBinData_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&offsetExData_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&label_offset_),
               sizeof(size_t));  // possible error: (original->IDType)
    input.read(reinterpret_cast<char *>(&size_data_per_element_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&size_links_per_element_), sizeof(size_t));

    input.read(reinterpret_cast<char *>(&maxlevel_), sizeof(int));
    input.read(reinterpret_cast<char *>(&enterpoint_node_), sizeof(IDType));

    input.read(reinterpret_cast<char *>(&M_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&maxM_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&maxM0_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&mult_), sizeof(double));
    input.read(reinterpret_cast<char *>(&ef_construction_), sizeof(size_t));

    centroids_memory_ =
        reinterpret_cast<char *>(malloc(num_cluster_ * padded_dim_ * sizeof(DataType)));

    input.read(centroids_memory_, num_cluster_ * padded_dim_ * sizeof(DataType));

    data_level0_memory_ = reinterpret_cast<char *>(malloc(max_elements_ * size_data_per_element_));

    input.read(data_level0_memory_, cur_element_count_ * size_data_per_element_);

    std::cout << "cur_element_count = " << cur_element_count_ << '\n';

    std::vector<std::mutex>(max_elements_).swap(link_list_locks_);
    std::vector<std::mutex>(kMaxLabelOperationLock).swap(label_op_locks_);

    linkLists_ = reinterpret_cast<char **>(malloc(sizeof(void *) * max_elements_));
    if (linkLists_ == nullptr) {
      throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
    }

    element_levels_ = std::vector<int>(max_elements_);
    revSize_ = 1.0 / mult_;
    ef_ = 10;

    for (size_t i = 0; i < cur_element_count_; i++) {
      label_lookup_[get_external_label(i)] = i;
      unsigned int link_list_size;
      input.read(reinterpret_cast<char *>(&link_list_size), sizeof(unsigned int));
      if (link_list_size == 0) {
        element_levels_[i] = 0;
        linkLists_[i] = nullptr;
      } else {
        element_levels_[i] = static_cast<int>(link_list_size / size_links_per_element_);
        linkLists_[i] = reinterpret_cast<char *>(malloc(link_list_size));
        if (linkLists_[i] == nullptr) {
          throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
        }
        input.read(linkLists_[i], link_list_size);
      }
    }

    visited_list_pool_ = std::make_unique<VisitedListPool>(1, max_elements_);

    input.close();

    this->query_config_ =
        alaya::rabitq_impl::faster_config(padded_dim_, SplitSingleQuery<DataType>::kNumBits);

    is_valid = true;
  }

  void construct(size_t cluster_num, const DataType *centroids, size_t data_num,
                 const DataType *data, IDType *cluster_ids, size_t num_threads = 0,
                 bool faster = false) {
    num_cluster_ = cluster_num;
    centroids_memory_ =
        reinterpret_cast<char *>(malloc(num_cluster_ * padded_dim_ * sizeof(DataType)));
    if (centroids_memory_ == nullptr) {
      throw std::runtime_error("Not enough memory: HNSW failed to allocate centroids");
    }

    for (size_t i = 0; i < cluster_num; ++i) {
      space_->rotate_data(centroids + (i * dim_),
                          reinterpret_cast<DataType *>(centroids_memory_) + (i * padded_dim_));
    }

    rabitq_impl::RabitqConfig config;
    if (faster) {
      config = rabitq_impl::faster_config(padded_dim_, ex_bits_ + 1);
    }

    std::cout << "Start HierarchicalNSW construction..." << '\n';
    rawDataPtr_ = data;
    std::cout << "Build edges with non-quantized vectors..." << '\n';
    parallel_for(0, data_num, num_threads, [&](size_t idx, size_t /*threadId*/) {
      add_point(idx, cluster_ids[idx], config);
    });
  }

  std::vector<std::vector<std::pair<DistanceType, IDType>>> search(const float *queries,
                                                                   size_t query_num, size_t TOPK,
                                                                   size_t efSearch,
                                                                   size_t thread_num) {
    set_ef(efSearch);
    std::vector<std::vector<std::pair<DistanceType, IDType>>> results(query_num);
    parallel_for(0, query_num, thread_num, [&](size_t idx, size_t /*threadId*/) {
      std::vector<float> rotated_query(padded_dim_);
      space_->rotate_data(queries + (idx * dim_), rotated_query.data());
      maxheap<std::pair<float, PID>> knn = search_knn(rotated_query.data(), TOPK);
      while (knn.size()) {
        results[idx].emplace_back(knn.top());
        knn.pop();
      }
      std::reverse(results[idx].begin(), results[idx].end());
    });
    return results;
  }
};
}  // namespace alaya
// NOLINTEND