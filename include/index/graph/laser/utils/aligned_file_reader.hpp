// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file aligned_file_reader.hpp
 * @brief Backend-neutral aligned file reader interface for LASER disk access.
 *
 * This module provides the common reader interface used by LASER search and
 * build code. Linux production builds use a libaio implementation; portable
 * builds provide the same interface through a thread-pool backend.
 *
 * Key Design Decisions:
 * - Keeps concrete backend APIs out of QG consumer code
 * - Preserves the 512-byte aligned read contract for all backends
 * - Requires 512-byte alignment for all buffers, offsets, and lengths
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cstdio>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
// #include "tsl/robin_map.h"
#if (defined(ALAYA_LASER_USE_LIBAIO) + defined(ALAYA_LASER_USE_THREADPOOL) + \
     defined(ALAYA_LASER_USE_IOCP)) > 1
  #error "Define only one LASER I/O backend macro"
#endif

#if !defined(ALAYA_LASER_USE_LIBAIO) && !defined(ALAYA_LASER_USE_THREADPOOL) && \
    !defined(ALAYA_LASER_USE_IOCP)
  #if defined(_WIN32)
    #define ALAYA_LASER_USE_IOCP 1
  #elif defined(__linux__)
    #define ALAYA_LASER_USE_LIBAIO 1
  #else
    #define ALAYA_LASER_USE_THREADPOOL 1
  #endif
#endif

#if defined(ALAYA_LASER_USE_LIBAIO)
  #include <fcntl.h>
  #include <libaio.h>
  #include <unistd.h>
#endif

#define MAX_EVENTS 1024

#ifndef ROUND_UP
  #define ROUND_UP(X, Y) \
    (((static_cast<uint64_t>(X) / (Y)) + (static_cast<uint64_t>(X) % (Y) != 0)) * (Y))
#endif  // !ROUND_UP

// alignment tests — IS_ALIGNED is invoked with both void* and integer arguments,
// so the only universally-correct cast is the C-style cast (which picks
// reinterpret_cast for pointers and static_cast for integers).
// NOLINTNEXTLINE(readability/casting)
#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)

#define MAX_IO_DEPTH 128

#if defined(ALAYA_LASER_USE_LIBAIO)
using IOContext = io_context_t;
#elif defined(ALAYA_LASER_USE_IOCP)
struct IOCPContext;
using IOContext = IOCPContext *;
#else
struct ThreadPoolContext;
using IOContext = ThreadPoolContext *;
#endif

/**
 * @brief Represents a single aligned read request for O_DIRECT I/O.
 *
 * All fields must be 512-byte aligned to satisfy O_DIRECT requirements.
 * This alignment constraint comes from the Linux kernel's direct I/O path,
 * which bypasses the page cache and requires sector-aligned access.
 */
struct AlignedRead {
  uint64_t offset;  // File offset to read from (must be 512-aligned)
  uint64_t len;     // Number of bytes to read (must be 512-aligned)
  uint64_t id;      // User-defined ID for tracking this request
  void *buf;        // Destination buffer (must be 512-aligned)

  AlignedRead() : offset(0), len(0), id(0), buf(nullptr) {}

  AlignedRead(uint64_t offset, uint64_t len, uint64_t id, void *buf)
      : offset(offset), len(len), id(id), buf(buf) {
    assert(IS_512_ALIGNED(offset));
    assert(IS_512_ALIGNED(len));
    assert(IS_512_ALIGNED(buf));
  }
};

struct AlignedReadEvent {
  uint64_t id = 0;
  int64_t result = 0;
};

class AlignedFileReader {
 protected:
  std::map<std::thread::id, IOContext> ctx_map_;
  std::mutex ctx_mut_;

 public:
  // Returns the thread-specific context owned by the reader. The returned
  // reference is valid until the calling thread is deregistered.
  virtual IOContext &get_ctx() = 0;

  virtual ~AlignedFileReader() {}

  // register thread-id for a context
  virtual void register_thread() = 0;
  // de-register thread-id for a context
  virtual void deregister_thread() = 0;
  virtual void deregister_all_threads() = 0;

  // Open & close ops
  // Blocking calls
  virtual void open(const std::string &fname) = 0;
  virtual void close() = 0;

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  virtual void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async = false) = 0;
  virtual int submit_reqs(std::vector<AlignedRead> &read_reqs, IOContext &ctx) = 0;
  // Blocking completion wait. Returns after n_ops completions have been
  // collected for ctx and appends backend-independent events to out.
  virtual int get_events(IOContext &ctx, int n_ops, std::vector<AlignedReadEvent> &out) = 0;

  void get_events(IOContext &ctx, int n_ops) {
    std::vector<AlignedReadEvent> ignored;
    (void)get_events(ctx, n_ops, ignored);
  }

  // Non-blocking completion poll. Returns immediately with up to max_events
  // completions; zero is valid and means callers should try again later.
  // Implementations must not throw when no completions are ready, but may throw
  // for genuine backend errors such as an invalid context.
  virtual int poll_events(IOContext &ctx, int max_events, std::vector<AlignedReadEvent> &out) = 0;
};

#if defined(ALAYA_LASER_USE_LIBAIO)

class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t file_sz_;
  int file_desc_;
  io_context_t bad_ctx_ = nullptr;

 public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader() override;

  IOContext &get_ctx() override;

  // register thread-id for a context
  void register_thread() override;

  // de-register thread-id for a context
  void deregister_thread() override;
  void deregister_all_threads() override;

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname) override;
  void close() override;

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async = false) override;

  int submit_reqs(std::vector<AlignedRead> &read_reqs, IOContext &ctx) override;
  int get_events(IOContext &ctx, int n_ops, std::vector<AlignedReadEvent> &out) override;
  int poll_events(IOContext &ctx, int max_events, std::vector<AlignedReadEvent> &out) override;
};

using io_event_t = struct io_event;
using iocb_t = struct iocb;

inline void execute_io(io_context_t ctx,
                       int fd,
                       std::vector<AlignedRead> &read_reqs,
                       uint64_t n_retries = 0) {
  #ifdef DEBUG
  for (auto &req : read_reqs) {
    assert(IS_ALIGNED(req.len, 512));
    assert(IS_ALIGNED(req.offset, 512));
    assert(IS_ALIGNED(req.buf, 512));
  }
  #endif

  // break-up requests into chunks of size MAX_EVENTS each
  uint64_t n_iters = ROUND_UP(read_reqs.size(), MAX_EVENTS) / MAX_EVENTS;
  for (uint64_t iter = 0; iter < n_iters; iter++) {
    uint64_t n_ops = std::min(static_cast<uint64_t>(read_reqs.size()) - (iter * MAX_EVENTS),
                              static_cast<uint64_t>(MAX_EVENTS));
    std::vector<iocb_t *> cbs(n_ops, nullptr);
    std::vector<io_event_t> evts(n_ops);
    std::vector<struct iocb> cb(n_ops);
    for (uint64_t j = 0; j < n_ops; j++) {
      io_prep_pread(cb.data() + j,
                    fd,
                    read_reqs[j + iter * MAX_EVENTS].buf,
                    read_reqs[j + iter * MAX_EVENTS].len,
                    static_cast<off_t>(read_reqs[j + iter * MAX_EVENTS].offset));
    }

    // initialize `cbs` using `cb` array
    for (uint64_t i = 0; i < n_ops; i++) {
      cbs[i] = cb.data() + i;
    }

    uint64_t n_tries = 0;
    while (n_tries <= n_retries) {
      // issue reads
      int64_t ret = io_submit(ctx, static_cast<int64_t>(n_ops), cbs.data());
      // if requests didn't get accepted
      if (ret != static_cast<int64_t>(n_ops)) {
        throw std::runtime_error("LinuxAlignedFileReader: io_submit() failed; returned " +
                                 std::to_string(ret) + ", expected=" + std::to_string(n_ops) +
                                 ", errno=" + std::to_string(errno) + "=" +
                                 ::strerror(static_cast<int>(-ret)));
      } else {
        // wait on io_getevents
        ret = io_getevents(ctx,
                           static_cast<int64_t>(n_ops),
                           static_cast<int64_t>(n_ops),
                           evts.data(),
                           nullptr);
        // if requests didn't complete
        if (ret != static_cast<int64_t>(n_ops)) {
          throw std::runtime_error("LinuxAlignedFileReader: io_getevents() failed; returned " +
                                   std::to_string(ret) + ", expected=" + std::to_string(n_ops) +
                                   ", errno=" + std::to_string(errno) + "=" +
                                   ::strerror(static_cast<int>(-ret)));
        } else {
          break;
        }
      }
    }
  }
}

inline LinuxAlignedFileReader::LinuxAlignedFileReader() { this->file_desc_ = -1; }

inline LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc_, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc_);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":"
                  << ::strerror(errno) << std::endl;
      }
    }
  }
}

inline io_context_t &LinuxAlignedFileReader::get_ctx() {
  std::unique_lock<std::mutex> lk(ctx_mut_);
  // perform checks only in DEBUG mode
  if (ctx_map_.find(std::this_thread::get_id()) == ctx_map_.end()) {
    std::cerr << "bad thread access; returning -1 as io_context_t" << std::endl;
    return this->bad_ctx_;
  }
  return ctx_map_[std::this_thread::get_id()];
}

inline void LinuxAlignedFileReader::register_thread() {
  auto my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut_);
  if (ctx_map_.find(my_id) != ctx_map_.end()) {
    std::cerr << "multiple calls to register_thread from the same thread" << std::endl;
    return;
  }
  io_context_t ctx = nullptr;
  int ret = io_setup(MAX_EVENTS, &ctx);
  if (ret != 0) {
    lk.unlock();
    assert(errno != EAGAIN);
    assert(errno != ENOMEM);
    std::cerr << "io_setup() failed; returned " << ret << ", errno=" << errno << ":"
              << ::strerror(errno) << std::endl;
  } else {
    ctx_map_[my_id] = ctx;
  }
  lk.unlock();
}

inline void LinuxAlignedFileReader::deregister_thread() {
  auto my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut_);
  assert(ctx_map_.find(my_id) != ctx_map_.end());

  lk.unlock();
  io_context_t ctx = this->get_ctx();
  io_destroy(ctx);
  lk.lock();
  ctx_map_.erase(my_id);
  lk.unlock();
}

inline void LinuxAlignedFileReader::deregister_all_threads() {
  std::unique_lock<std::mutex> lk(ctx_mut_);
  for (auto &x : ctx_map_) {
    io_context_t ctx = x.second;
    io_destroy(ctx);
  }
  ctx_map_.clear();
}

inline void LinuxAlignedFileReader::open(const std::string &fname) {
  close();
  // O_DIRECT: Bypass the OS page cache and read directly from disk.
  //   Why? For large-scale vector search, the working set far exceeds RAM.
  //   Without O_DIRECT, the OS would try to cache data, causing:
  //   1. Unpredictable memory usage (cache competes with our application memory)
  //   2. Double-buffering overhead (data copied: disk -> page cache -> user buffer)
  //   3. Cache thrashing (random access patterns pollute the page cache)
  //   With O_DIRECT, we have full control over memory and avoid these issues.
  //
  // O_RDONLY: Open file for reading only (index files are read-only during search).
  int flags = O_DIRECT | O_RDONLY;
  this->file_desc_ = ::open(fname.c_str(), flags);
  assert(this->file_desc_ != -1);
}

inline void LinuxAlignedFileReader::close() {
  if (this->file_desc_ == -1) {
    return;
  }
  ::fcntl(this->file_desc_, F_GETFD);
  ::close(this->file_desc_);
  this->file_desc_ = -1;
}

inline void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                         io_context_t &ctx,
                                         bool async) {
  if (async) {
    std::cout << "Async currently not supported in linux." << std::endl;
  }
  assert(this->file_desc_ != -1);
  execute_io(ctx, this->file_desc_, read_reqs);
}

/**
 * @brief Submit read requests asynchronously (non-blocking).
 *
 * Submits I/O requests to the kernel AIO queue and returns immediately
 * without waiting for completion. Use io_getevents() to check for completion.
 * This enables overlapping disk I/O with CPU computation for better throughput.
 *
 * @param read_reqs Vector of aligned read requests to submit
 * @param ctx       Thread-local AIO context
 * @return Number of requests successfully submitted
 */
inline int LinuxAlignedFileReader::submit_reqs(std::vector<AlignedRead> &read_reqs,
                                               io_context_t &ctx) {
  assert(this->file_desc_ != -1);

  if (read_reqs.size() > MAX_EVENTS) {
    throw std::runtime_error("LinuxAlignedFileReader::submit_reqs: request count " +
                             std::to_string(read_reqs.size()) +
                             " exceeds MAX_EVENTS=" + std::to_string(MAX_EVENTS));
  }
  size_t n_ops = read_reqs.size();
  std::vector<struct iocb *> cbs(n_ops, nullptr);
  std::vector<io_event> evts(n_ops);
  std::vector<struct iocb> cb(n_ops);
  for (size_t j = 0; j < n_ops; j++) {
    io_prep_pread(cb.data() + j,
                  this->file_desc_,
                  read_reqs[j].buf,
                  read_reqs[j].len,
                  static_cast<off_t>(read_reqs[j].offset));
  }
  for (size_t i = 0; i < n_ops; i++) {
    cbs[i] = cb.data() + i;
    cbs[i]->data = reinterpret_cast<void *>(static_cast<uintptr_t>(read_reqs[i].id));
    {
    }
  }

  int ret = io_submit(ctx, static_cast<int64_t>(n_ops), cbs.data());
  if (ret != static_cast<int>(n_ops)) {
    throw std::runtime_error("LinuxAlignedFileReader::submit_reqs: io_submit() failed; returned " +
                             std::to_string(ret) + ", expected=" + std::to_string(n_ops) +
                             ", errno=" + std::to_string(errno) + "=" + ::strerror(-ret));
  }
  return static_cast<int>(n_ops);
}

inline int LinuxAlignedFileReader::get_events(IOContext &ctx,
                                              int n_ops,
                                              std::vector<AlignedReadEvent> &out) {
  std::vector<io_event> evts(n_ops);
  auto ret = io_getevents(ctx,
                          static_cast<int64_t>(n_ops),
                          static_cast<int64_t>(n_ops),
                          evts.data(),
                          nullptr);
  if (ret != static_cast<int64_t>(n_ops)) {
    throw std::runtime_error(
        "LinuxAlignedFileReader::get_events: io_getevents() failed; returned " +
        std::to_string(ret));
  }
  out.clear();
  out.reserve(static_cast<size_t>(ret));
  for (int64_t i = 0; i < ret; ++i) {
    out.push_back({static_cast<uint64_t>(reinterpret_cast<uintptr_t>(evts[i].data)),
                   static_cast<int64_t>(evts[i].res)});
  }
  return static_cast<int>(ret);
}

inline int LinuxAlignedFileReader::poll_events(IOContext &ctx,
                                               int max_events,
                                               std::vector<AlignedReadEvent> &out) {
  out.clear();
  if (max_events <= 0) {
    return 0;
  }

  std::vector<io_event> evts(static_cast<size_t>(max_events));
  auto ret = io_getevents(ctx, 0, static_cast<int64_t>(max_events), evts.data(), nullptr);
  if (ret < 0) {
    throw std::runtime_error(
        "LinuxAlignedFileReader::poll_events: io_getevents() failed; returned " +
        std::to_string(ret) + ", errno=" + std::to_string(-ret));
  }

  out.reserve(static_cast<size_t>(ret));
  for (int64_t i = 0; i < ret; ++i) {
    out.push_back({static_cast<uint64_t>(reinterpret_cast<uintptr_t>(evts[i].data)),
                   static_cast<int64_t>(evts[i].res)});
  }
  return static_cast<int>(ret);
}

#endif  // defined(ALAYA_LASER_USE_LIBAIO)
