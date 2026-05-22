// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Windows-only IOCP backend smoke test. Mirrors the threadpool fixture's
// shape (page-aligned synthetic file, single+multi-thread sync/poll cycles,
// EOF short-read) so the matrix stays comparable across backends.

#include <gtest/gtest.h>

#ifdef _WIN32

  #include <algorithm>
  #include <cstdint>
  #include <cstring>
  #include <filesystem>
  #include <fstream>
  #include <mutex>
  #include <string>
  #include <thread>
  #include <vector>

  #include "index/graph/laser/utils/aligned_file_reader.hpp"
  #include "index/graph/laser/utils/iocp_file_reader.hpp"
  #include "index/graph/laser/utils/memory.hpp"

namespace {

constexpr size_t kPageSize = 4096;

class IOCPFileReaderTest : public ::testing::Test {
 protected:
  std::filesystem::path root_;
  std::filesystem::path file_path_;
  std::vector<char> file_bytes_;

  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("alaya_iocp_file_reader_" + std::to_string(::GetCurrentProcessId()));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);
    file_path_ = root_ / "aligned.bin";

    // 4 pages so we can read into [0, 1, 2, 3] and also test past-EOF at
    // offset 8*page (4-page hole expected).
    file_bytes_.resize(4 * kPageSize);
    for (size_t i = 0; i < file_bytes_.size(); ++i) {
      file_bytes_[i] = static_cast<char>((i * 17U + 3U) & 0x7F);
    }

    std::ofstream out(file_path_, std::ios::binary);
    out.write(file_bytes_.data(), static_cast<std::streamsize>(file_bytes_.size()));
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  static char *aligned_buffer(size_t bytes) {
    return reinterpret_cast<char *>(alaya::laser::memory::align_allocate<kPageSize>(bytes));
  }
};

TEST_F(IOCPFileReaderTest, OpenCloseIsIdempotent) {
  IOCPFileReader reader;

  reader.open(file_path_.string());
  reader.close();
  reader.close();
}

TEST_F(IOCPFileReaderTest, BlockingReadCompletesSingleThreadBatch) {
  IOCPFileReader reader;
  reader.open(file_path_.string());
  reader.register_thread();

  char *first = aligned_buffer(kPageSize);
  char *second = aligned_buffer(kPageSize);
  std::vector<AlignedRead> reads;
  reads.emplace_back(0, kPageSize, 10, first);
  reads.emplace_back(kPageSize, kPageSize, 11, second);

  reader.read(reads, reader.get_ctx());

  EXPECT_EQ(std::memcmp(first, file_bytes_.data(), kPageSize), 0);
  EXPECT_EQ(std::memcmp(second, file_bytes_.data() + kPageSize, kPageSize), 0);

  alaya::laser::memory::align_free(first);
  alaya::laser::memory::align_free(second);
  reader.deregister_thread();
  reader.close();
}

TEST_F(IOCPFileReaderTest, PollEventsReturnsCompletedIdsWithoutBlocking) {
  IOCPFileReader reader;
  reader.open(file_path_.string());
  reader.register_thread();

  char *first = aligned_buffer(kPageSize);
  char *second = aligned_buffer(kPageSize);
  std::vector<AlignedRead> reads;
  reads.emplace_back(0, kPageSize, 21, first);
  reads.emplace_back(2 * kPageSize, kPageSize, 22, second);

  ASSERT_EQ(reader.submit_reqs(reads, reader.get_ctx()), 2);

  std::vector<AlignedReadEvent> events;
  for (int attempts = 0; attempts < 1000 && events.size() < reads.size(); ++attempts) {
    std::vector<AlignedReadEvent> polled;
    const int count = reader.poll_events(reader.get_ctx(), 2, polled);
    ASSERT_GE(count, 0);
    events.insert(events.end(), polled.begin(), polled.end());
    if (events.size() < reads.size()) {
      std::this_thread::yield();
    }
  }

  ASSERT_EQ(events.size(), reads.size());
  std::vector<uint64_t> ids;
  std::transform(events.begin(), events.end(), std::back_inserter(ids), [](const auto &event) {
    return event.id;
  });
  std::sort(ids.begin(), ids.end());
  EXPECT_EQ(ids, (std::vector<uint64_t>{21, 22}));
  for (const auto &event : events) {
    EXPECT_EQ(event.result, static_cast<int64_t>(kPageSize));
  }
  EXPECT_EQ(std::memcmp(first, file_bytes_.data(), kPageSize), 0);
  EXPECT_EQ(std::memcmp(second, file_bytes_.data() + 2 * kPageSize, kPageSize), 0);

  alaya::laser::memory::align_free(first);
  alaya::laser::memory::align_free(second);
  reader.deregister_thread();
  reader.close();
}

TEST_F(IOCPFileReaderTest, SupportsIndependentCompletionQueuesPerThread) {
  IOCPFileReader reader;
  reader.open(file_path_.string());

  std::vector<uint64_t> completed_ids;
  std::mutex completed_mutex;
  auto worker = [&](uint64_t id, uint64_t offset) {
    reader.register_thread();
    char *buf = aligned_buffer(kPageSize);
    std::vector<AlignedRead> reads;
    reads.emplace_back(offset, kPageSize, id, buf);
    ASSERT_EQ(reader.submit_reqs(reads, reader.get_ctx()), 1);

    std::vector<AlignedReadEvent> events;
    ASSERT_EQ(reader.get_events(reader.get_ctx(), 1, events), 1);
    ASSERT_EQ(events.size(), 1U);
    EXPECT_EQ(events[0].id, id);
    EXPECT_EQ(events[0].result, static_cast<int64_t>(kPageSize));
    EXPECT_EQ(std::memcmp(buf, file_bytes_.data() + offset, kPageSize), 0);
    {
      std::lock_guard<std::mutex> lock(completed_mutex);
      completed_ids.push_back(events[0].id);
    }
    alaya::laser::memory::align_free(buf);
    reader.deregister_thread();
  };

  std::thread first(worker, 31, 0);
  std::thread second(worker, 32, kPageSize);
  first.join();
  second.join();

  std::sort(completed_ids.begin(), completed_ids.end());
  EXPECT_EQ(completed_ids, (std::vector<uint64_t>{31, 32}));
  reader.close();
}

TEST_F(IOCPFileReaderTest, ReportsShortReadAndRejectsClosedReader) {
  IOCPFileReader reader;
  reader.open(file_path_.string());
  reader.register_thread();

  char *buf = aligned_buffer(kPageSize);
  std::vector<AlignedRead> reads;
  // Read at 8*page (past EOF, file is only 4 pages). IOCP returns
  // ERROR_HANDLE_EOF which we map to result=0 (success, zero bytes).
  reads.emplace_back(8 * kPageSize, kPageSize, 41, buf);
  ASSERT_EQ(reader.submit_reqs(reads, reader.get_ctx()), 1);

  std::vector<AlignedReadEvent> events;
  ASSERT_EQ(reader.get_events(reader.get_ctx(), 1, events), 1);
  ASSERT_EQ(events.size(), 1U);
  EXPECT_EQ(events[0].id, 41U);
  EXPECT_EQ(events[0].result, 0);

  reader.close();
  EXPECT_THROW(reader.submit_reqs(reads, reader.get_ctx()), std::runtime_error);

  alaya::laser::memory::align_free(buf);
}

}  // namespace

#else

// Linux / macOS: nothing to test for the IOCP backend; emit a trivial
// passing test so the suite remains consistent across platforms.
TEST(IOCPFileReaderTest, NotApplicableOnNonWindowsPlatforms) { SUCCEED(); }

#endif  // _WIN32
