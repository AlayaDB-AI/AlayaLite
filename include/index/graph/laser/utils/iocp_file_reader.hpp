// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file iocp_file_reader.hpp
 * @brief Windows IOCP (I/O Completion Port) AlignedFileReader backend.
 *
 * Counterpart to LinuxAlignedFileReader (libaio) and ThreadPoolFileReader
 * (portable pread+threads). Topology:
 *
 *   - Single process-shared `HANDLE hCompletionPort` covering one file handle
 *     opened with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`.
 *   - Each registered consumer thread owns an `IOCPContext` with a
 *     moodycamel::ConcurrentQueue<AlignedReadEvent> for completions and a
 *     stable monotonic uint64 thread id that is passed as
 *     `CompletionKey` per file-handle association.
 *   - One dedicated dispatcher thread drains the port via
 *     `GetQueuedCompletionStatusEx` and routes each entry into the
 *     originating thread's queue by looking up `lpCompletionKey`.
 *
 * The Linux libaio builder uses synchronous `pwrite`, so this reader only
 * handles the search-side read path. Writes go through `platform_fs` helpers
 * elsewhere.
 *
 * Active when `ALAYA_LASER_USE_IOCP=1`; gated by `_WIN32` for header parsing.
 */

#pragma once

#ifdef _WIN32

  #include <algorithm>
  #include <atomic>
  #include <chrono>
  #include <cstddef>
  #include <cstdint>
  #include <cstdio>
  #include <cstring>
  #include <iostream>
  #include <memory>
  #include <mutex>
  #include <stdexcept>
  #include <string>
  #include <thread>
  #include <unordered_map>
  #include <utility>
  #include <vector>

  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>

  #include "concurrentqueue.h"  // NOLINT
  #include "index/graph/laser/utils/aligned_file_reader.hpp"

struct IOCPContext {
  // Stable monotonic id assigned by `register_thread()`. Passed as the
  // CompletionKey to CreateIoCompletionPort so the dispatcher can route
  // entries back to the owning thread's queue.
  uint64_t tid{0};
  moodycamel::ConcurrentQueue<AlignedReadEvent> completions;
  std::atomic<bool> active{true};
};

class IOCPFileReader : public AlignedFileReader {
 private:
  // OVERLAPPED extension to carry the AlignedRead.id and the owning
  // tid through the completion. Both are recovered by the dispatcher.
  // `OVERLAPPED` must remain the first member so a `OVERLAPPED*` can be
  // safely reinterpret_cast'd back to `ReadOverlapped*`.
  struct ReadOverlapped {
    OVERLAPPED ov{};
    uint64_t id{0};
    uint64_t tid{0};
  };

  HANDLE hFile_ = INVALID_HANDLE_VALUE;
  HANDLE hCompletionPort_ = nullptr;
  std::atomic<bool> stop_{false};
  std::atomic<bool> shutting_down_{false};
  // In-flight ReadOverlapped count. close() spins until this drops to zero
  // after CancelIoEx so no pending OVERLAPPED is leaked across shutdown.
  std::atomic<uint64_t> pending_ops_{0};
  std::atomic<uint64_t> next_tid_{1};
  std::thread dispatcher_;
  // Shared ownership for safe dispatcher-vs-deregister lifetime: dispatcher
  // holds a strong ref while enqueueing, so a concurrent deregister can't
  // free the IOCPContext mid-enqueue.
  std::unordered_map<std::thread::id, std::shared_ptr<IOCPContext>> owned_contexts_;
  std::unordered_map<uint64_t, std::shared_ptr<IOCPContext>> tid_to_ctx_;
  std::mutex tid_map_mut_;

  // 50 ms wakeup also bounds shutdown latency: stop_ is observed at most
  // one timeout later, even if the sentinel completion is lost.
  static constexpr DWORD kCompletionTimeoutMs = 50;
  static constexpr ULONG kCompletionBatchSize = 64;

  static auto win32_error_to_result(DWORD err) noexcept -> int64_t {
    // EOF maps to zero bytes transferred — caller treats as a normal short
    // read. All other errors surface as negative codes; the consumer treats
    // negative as a generic I/O failure.
    if (err == ERROR_HANDLE_EOF) {
      return 0;
    }
    return -static_cast<int64_t>(err);
  }

  void dispatcher_loop() {
    std::vector<OVERLAPPED_ENTRY> entries(kCompletionBatchSize);
    while (!stop_.load(std::memory_order_acquire)) {
      ULONG n_out = 0;
      const BOOL ok = ::GetQueuedCompletionStatusEx(hCompletionPort_,
                                                    entries.data(),
                                                    kCompletionBatchSize,
                                                    &n_out,
                                                    kCompletionTimeoutMs,
                                                    FALSE);
      if (!ok) {
        const DWORD err = ::GetLastError();
        if (err == WAIT_TIMEOUT) {
          continue;
        }
        if (err == ERROR_ABANDONED_WAIT_0 || err == ERROR_INVALID_HANDLE) {
          // Port was closed under us; bail out.
          break;
        }
        // Unexpected error — log and continue so we don't lose other
        // completions that may still arrive.
        std::cerr << "IOCPFileReader: GetQueuedCompletionStatusEx error " << err << std::endl;
        continue;
      }
      for (ULONG i = 0; i < n_out; ++i) {
        const auto &entry = entries[i];
        if (entry.lpOverlapped == nullptr) {
          // PostQueuedCompletionStatus shutdown sentinel (lpOverlapped null
          // and lpCompletionKey 0). Exit immediately.
          stop_.store(true, std::memory_order_release);
          break;
        }
        auto *ro = reinterpret_cast<ReadOverlapped *>(entry.lpOverlapped);
        const uint64_t tid = ro->tid;
        AlignedReadEvent ev{};
        ev.id = ro->id;
        // Inspect OVERLAPPED.Internal (NTSTATUS) for true success/failure
        // — zero bytes is a legitimate EOF success, not an error to be
        // distinguished. NTSTATUS 0 == STATUS_SUCCESS; non-zero means
        // map via GetOverlappedResult or directly to the Win32 error.
        const ULONG_PTR nt_status = entry.lpOverlapped->Internal;
        if (nt_status == 0) {
          ev.result = static_cast<int64_t>(entry.dwNumberOfBytesTransferred);
        } else {
          DWORD bytes = 0;
          if (::GetOverlappedResult(hFile_, entry.lpOverlapped, &bytes, FALSE)) {
            ev.result = static_cast<int64_t>(bytes);
          } else {
            ev.result = win32_error_to_result(::GetLastError());
          }
        }
        // Hold a strong ref to the IOCPContext for the duration of the
        // enqueue so a concurrent deregister can't free the underlying
        // queue object.
        std::shared_ptr<IOCPContext> ctx_ref;
        {
          std::lock_guard<std::mutex> lk(tid_map_mut_);
          auto it = tid_to_ctx_.find(tid);
          if (it != tid_to_ctx_.end()) {
            ctx_ref = it->second;
          }
        }
        if (ctx_ref) {
          ctx_ref->completions.enqueue(ev);
        }
        // Always delete the OVERLAPPED — it's the per-submit allocation,
        // and ownership transferred to the kernel until completion lands.
        delete ro;
        pending_ops_.fetch_sub(1, std::memory_order_acq_rel);
      }
    }
  }

 public:
  IOCPFileReader() = default;
  ~IOCPFileReader() override { close(); }

  IOCPFileReader(const IOCPFileReader &) = delete;
  auto operator=(const IOCPFileReader &) -> IOCPFileReader & = delete;
  IOCPFileReader(IOCPFileReader &&) = delete;
  auto operator=(IOCPFileReader &&) -> IOCPFileReader & = delete;

  auto get_ctx() -> IOContext & override {
    std::unique_lock<std::mutex> lk(ctx_mut_);
    auto it = ctx_map_.find(std::this_thread::get_id());
    if (it == ctx_map_.end()) {
      throw std::runtime_error(
          "IOCPFileReader::get_ctx: calling thread is not registered "
          "(call register_thread() first)");
    }
    return it->second;
  }

  void register_thread() override {
    const auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut_);
    if (ctx_map_.find(my_id) != ctx_map_.end()) {
      throw std::runtime_error("IOCPFileReader::register_thread: thread is already registered");
    }
    auto ctx = std::make_shared<IOCPContext>();
    ctx->tid = next_tid_.fetch_add(1, std::memory_order_relaxed);
    IOContext raw = ctx.get();
    {
      std::lock_guard<std::mutex> lk_tid(tid_map_mut_);
      tid_to_ctx_[ctx->tid] = ctx;
    }
    owned_contexts_[my_id] = std::move(ctx);
    ctx_map_[my_id] = raw;
  }

  void deregister_thread() override {
    const auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut_);
    auto own_it = owned_contexts_.find(my_id);
    if (own_it == owned_contexts_.end()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lk_tid(tid_map_mut_);
      tid_to_ctx_.erase(own_it->second->tid);
    }
    // The dispatcher may still hold a temporary shared_ptr ref; this
    // shared ownership keeps the IOCPContext alive until the in-flight
    // enqueue completes.
    own_it->second->active.store(false, std::memory_order_release);
    ctx_map_.erase(my_id);
    owned_contexts_.erase(my_id);
  }

  void deregister_all_threads() override {
    std::unique_lock<std::mutex> lk(ctx_mut_);
    {
      std::lock_guard<std::mutex> lk_tid(tid_map_mut_);
      tid_to_ctx_.clear();
    }
    for (auto &entry : owned_contexts_) {
      entry.second->active.store(false, std::memory_order_release);
    }
    ctx_map_.clear();
    owned_contexts_.clear();
  }

  void open(const std::string &fname) override {
    close();

    // Convert the narrow path to wide-char for CreateFileW. `fname` is
    // assumed ASCII: MSVC's `std::filesystem::path::string()` uses
    // CP_THREAD_ACP, not UTF-8, so a non-ASCII path would be decoded
    // incorrectly here. Non-ASCII paths on Windows require extending
    // the AlignedFileReader interface to accept `std::filesystem::path`
    // and forwarding `path::c_str()` (wide) directly.
    int wide_len = ::MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, nullptr, 0);
    if (wide_len <= 0) {
      throw std::runtime_error("IOCPFileReader::open: invalid UTF-8 in path: " + fname);
    }
    std::wstring wname(static_cast<size_t>(wide_len), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, wname.data(), wide_len);
    // Strip the trailing NUL that MultiByteToWideChar wrote.
    if (!wname.empty() && wname.back() == L'\0') {
      wname.pop_back();
    }

    hFile_ = ::CreateFileW(wname.c_str(),
                           GENERIC_READ,
                           FILE_SHARE_READ,
                           nullptr,
                           OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
                           nullptr);
    if (hFile_ == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("IOCPFileReader::open: CreateFileW failed for " + fname +
                               ": Win32 error " + std::to_string(::GetLastError()));
    }

    // `CreateIoCompletionPort` allows only one (handle, key) association, so
    // the per-handle CompletionKey is uniform and cannot route per-thread.
    // Per-read routing is carried in the `ReadOverlapped::tid` field instead.
    hCompletionPort_ = ::CreateIoCompletionPort(hFile_, nullptr, 0, 0);
    if (hCompletionPort_ == nullptr) {
      const auto err = ::GetLastError();
      ::CloseHandle(hFile_);
      hFile_ = INVALID_HANDLE_VALUE;
      throw std::runtime_error("IOCPFileReader::open: CreateIoCompletionPort failed: Win32 error " +
                               std::to_string(err));
    }

    stop_.store(false, std::memory_order_release);
    dispatcher_ = std::thread([this]() {
      dispatcher_loop();
    });
  }

  void close() override {
    if (!dispatcher_.joinable() && hFile_ == INVALID_HANDLE_VALUE) {
      return;
    }
    // Two-phase shutdown to avoid leaking ReadOverlapped allocations:
    //   1. Mark shutting down so submit_reqs starts rejecting.
    //   2. CancelIoEx all in-flight reads on the file handle. Cancelled
    //      reads still produce a completion packet with STATUS_CANCELLED,
    //      so the dispatcher will drain them and delete the OVERLAPPEDs.
    //   3. Wait until pending_ops_ reaches zero. Bounded wait since
    //      CancelIoEx completion is guaranteed to land.
    //   4. Post the sentinel + join + close.
    shutting_down_.store(true, std::memory_order_release);
    if (hFile_ != INVALID_HANDLE_VALUE) {
      (void)::CancelIoEx(hFile_, nullptr);
    }
    // Bounded drain — give the dispatcher up to 5 seconds to clear the
    // cancellation completions. In practice the kernel posts cancellations
    // within microseconds; this cap just stops the destructor from hanging
    // forever on a borked handle.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (pending_ops_.load(std::memory_order_acquire) > 0 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (dispatcher_.joinable()) {
      stop_.store(true, std::memory_order_release);
      if (hCompletionPort_ != nullptr) {
        ::PostQueuedCompletionStatus(hCompletionPort_, 0, 0, nullptr);
      }
      dispatcher_.join();
    }
    if (hCompletionPort_ != nullptr) {
      ::CloseHandle(hCompletionPort_);
      hCompletionPort_ = nullptr;
    }
    if (hFile_ != INVALID_HANDLE_VALUE) {
      ::CloseHandle(hFile_);
      hFile_ = INVALID_HANDLE_VALUE;
    }
    deregister_all_threads();
    shutting_down_.store(false, std::memory_order_release);
  }

  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async = false) override {
    const int submitted = submit_reqs(read_reqs, ctx);
    if (async) {
      return;
    }
    std::vector<AlignedReadEvent> drained;
    const int completed = get_events(ctx, submitted, drained);
    if (completed != submitted) {
      throw std::runtime_error("IOCPFileReader::read: completed " + std::to_string(completed) +
                               " reads, expected " + std::to_string(submitted));
    }
  }

  int submit_reqs(std::vector<AlignedRead> &read_reqs, IOContext &ctx) override {
    if (hFile_ == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("IOCPFileReader::submit_reqs: reader is not open");
    }
    if (ctx == nullptr) {
      throw std::runtime_error("IOCPFileReader::submit_reqs: invalid thread context");
    }
    if (shutting_down_.load(std::memory_order_acquire)) {
      throw std::runtime_error("IOCPFileReader::submit_reqs: reader is shutting down");
    }
    if (read_reqs.size() > MAX_EVENTS) {
      throw std::runtime_error("IOCPFileReader::submit_reqs: request count " +
                               std::to_string(read_reqs.size()) +
                               " exceeds MAX_EVENTS=" + std::to_string(MAX_EVENTS));
    }

    auto *typed_ctx = static_cast<IOCPContext *>(ctx);
    int submitted = 0;
    for (auto &req : read_reqs) {
      auto *ro = new ReadOverlapped();
      ro->id = req.id;
      ro->tid = typed_ctx->tid;
      ro->ov.Offset = static_cast<DWORD>(req.offset & 0xFFFFFFFFULL);
      ro->ov.OffsetHigh = static_cast<DWORD>((req.offset >> 32) & 0xFFFFFFFFULL);
      // CreateIoCompletionPort associations are one-per-handle, so the
      // CompletionKey on the OVERLAPPED_ENTRY is uniform (0). Per-read tid
      // routing therefore lives in the ReadOverlapped wrapper, which the
      // dispatcher casts back from `entry.lpOverlapped`.

      // Pre-increment pending_ops_ so the dispatcher's decrement is balanced
      // even when ReadFile fails synchronously without queuing a completion.
      pending_ops_.fetch_add(1, std::memory_order_acq_rel);
      DWORD bytes_read = 0;
      const BOOL ok =
          ::ReadFile(hFile_, req.buf, static_cast<DWORD>(req.len), &bytes_read, &ro->ov);
      if (!ok) {
        const DWORD err = ::GetLastError();
        if (err != ERROR_IO_PENDING) {
          delete ro;
          pending_ops_.fetch_sub(1, std::memory_order_acq_rel);
          throw std::runtime_error("IOCPFileReader::submit_reqs: ReadFile failed at offset " +
                                   std::to_string(req.offset) + ", len " + std::to_string(req.len) +
                                   ": Win32 error " + std::to_string(err));
        }
        // ERROR_IO_PENDING — async completion will arrive via the port.
      }
      // Synchronous completion (ok == TRUE) also delivers a packet to the
      // port by default (FILE_SKIP_COMPLETION_PORT_ON_SUCCESS not set), so
      // we don't enqueue here — the dispatcher will see it like any other
      // completion and decrement pending_ops_.
      ++submitted;
    }
    return submitted;
  }

  int get_events(IOContext &ctx, int n_ops, std::vector<AlignedReadEvent> &out) override {
    out.clear();
    if (n_ops <= 0) {
      return 0;
    }
    if (ctx == nullptr) {
      throw std::runtime_error("IOCPFileReader::get_events: invalid thread context");
    }
    auto *typed_ctx = static_cast<IOCPContext *>(ctx);
    out.reserve(static_cast<size_t>(n_ops));
    while (static_cast<int>(out.size()) < n_ops) {
      AlignedReadEvent event;
      while (static_cast<int>(out.size()) < n_ops && typed_ctx->completions.try_dequeue(event)) {
        out.push_back(event);
      }
      if (static_cast<int>(out.size()) >= n_ops) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return static_cast<int>(out.size());
  }

  int poll_events(IOContext &ctx, int max_events, std::vector<AlignedReadEvent> &out) override {
    out.clear();
    if (max_events <= 0) {
      return 0;
    }
    if (ctx == nullptr) {
      throw std::runtime_error("IOCPFileReader::poll_events: invalid thread context");
    }
    auto *typed_ctx = static_cast<IOCPContext *>(ctx);
    out.resize(static_cast<size_t>(max_events));
    const size_t count =
        typed_ctx->completions.try_dequeue_bulk(out.data(), static_cast<size_t>(max_events));
    out.resize(count);
    return static_cast<int>(count);
  }
};

#endif  // _WIN32
