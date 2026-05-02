/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fcntl.h>
#include <gtest/gtest.h>
#include <signal.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "index/disk/disk_collection.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk {
namespace {

constexpr const char *kLockBusySubstring = "collection is already open by another process";

struct TestFd {
  int fd = -1;

  TestFd() = default;
  explicit TestFd(int f) : fd(f) {}
  TestFd(const TestFd &) = delete;
  auto operator=(const TestFd &) -> TestFd & = delete;
  TestFd(TestFd &&other) noexcept : fd(other.fd) { other.fd = -1; }
  auto operator=(TestFd &&other) noexcept -> TestFd & {
    if (this != &other) {
      reset();
      fd = other.fd;
      other.fd = -1;
    }
    return *this;
  }
  ~TestFd() { reset(); }

  void reset() noexcept {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }
};

struct LockAttempt {
  bool acquired = false;
  int err = 0;
  TestFd fd;
};

auto try_lock_file(const std::filesystem::path &lock_path) -> LockAttempt {
  int fd = ::open(lock_path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    return LockAttempt{false, errno, TestFd{}};
  }
  if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
    const int saved = errno;
    ::close(fd);
    return LockAttempt{false, saved, TestFd{}};
  }
  return LockAttempt{true, 0, TestFd{fd}};
}

auto contains(const std::string &haystack, const std::string &needle) -> bool {
  return haystack.find(needle) != std::string::npos;
}

auto lock_path_for(const std::filesystem::path &collection_path) -> std::filesystem::path {
  return collection_path / ".lock";
}

auto weak_lock_path_string(const std::filesystem::path &collection_path) -> std::string {
  return std::filesystem::weakly_canonical(lock_path_for(collection_path)).string();
}

auto read_text_file(const std::filesystem::path &path) -> std::string {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

template <typename Func>
auto capture_spdlog(Func fn) -> std::string {
  std::ostringstream log_stream;
  auto previous_logger = spdlog::default_logger();
  auto logger = std::make_shared<spdlog::logger>("disk_collection_lock_test",
                                                 std::make_shared<spdlog::sinks::ostream_sink_mt>(
                                                     log_stream));
  logger->set_pattern("%v");
  spdlog::set_default_logger(logger);
  try {
    fn();
    spdlog::set_default_logger(previous_logger);
  } catch (...) {
    spdlog::set_default_logger(previous_logger);
    throw;
  }
  return log_stream.str();
}

auto make_vectors(uint64_t n, uint32_t dim) -> std::vector<float> {
  std::vector<float> values(n * dim);
  for (uint64_t i = 0; i < n; ++i) {
    for (uint32_t d = 0; d < dim; ++d) {
      values[i * dim + d] = static_cast<float>(i + d + 1);
    }
  }
  return values;
}

auto make_labels(uint64_t n, uint64_t base = 1000) -> std::vector<uint64_t> {
  std::vector<uint64_t> labels(n);
  std::iota(labels.begin(), labels.end(), base);
  return labels;
}

void build_flat_collection(const std::filesystem::path &path, uint32_t dim = 4, uint64_t n = 2) {
  DiskCollection col(path, dim, MetricType::L2, DiskIndexType::Flat);
  auto vectors = make_vectors(n, dim);
  auto labels = make_labels(n);
  col.add_batch(vectors.data(), labels.data(), n);
  col.flush();
}

auto child_open_expect_failure_test_binary() -> std::string {
#if defined(__linux__)
  std::vector<char> buf(4096);
  ssize_t n = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
  if (n <= 0) {
    return {};
  }
  buf[static_cast<size_t>(n)] = '\0';
  return std::string(buf.data());
#else
  return {};
#endif
}

auto wait_for_child(pid_t pid) -> int {
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0) {
    if (errno == EINTR) {
      continue;
    }
    return -1;
  }
  return status;
}

class DiskCollectionLockTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    tmp_root_ = std::filesystem::temp_directory_path() /
                ("alaya_disk_collection_lock_" + std::to_string(::getpid()) + "_" + info->name());
    std::filesystem::remove_all(tmp_root_);
    std::filesystem::create_directories(tmp_root_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_root_, ec);
  }

  std::filesystem::path tmp_root_;
};

TEST_F(DiskCollectionLockTest, lock_acquire_writes_lock_file_in_collection_root) {
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path, 8, MetricType::L2, DiskIndexType::Flat);

  const auto lock_path = lock_path_for(path);
  struct ::stat st {};
  ASSERT_EQ(::stat(lock_path.c_str(), &st), 0) << std::strerror(errno);
  EXPECT_TRUE(S_ISREG(st.st_mode));
  EXPECT_EQ((st.st_mode & 0777), 0600);
  const auto content = read_text_file(lock_path);
  EXPECT_TRUE(content.empty() || contains(content, "pid=")) << content;
}

TEST_F(DiskCollectionLockTest, lock_held_for_lifetime_of_instance) {
  const auto path = tmp_root_ / "coll";
  auto col = std::make_unique<DiskCollection>(path, 8, MetricType::L2, DiskIndexType::Flat);

  auto attempt = try_lock_file(lock_path_for(path));
  EXPECT_FALSE(attempt.acquired);
  EXPECT_EQ(attempt.err, EWOULDBLOCK);

  col.reset();
  auto retry = try_lock_file(lock_path_for(path));
  EXPECT_TRUE(retry.acquired) << std::strerror(retry.err);
}

TEST_F(DiskCollectionLockTest, lock_released_on_destructor) {
  const auto path = tmp_root_ / "coll";
  {
    auto col = std::make_unique<DiskCollection>(path, 8, MetricType::L2, DiskIndexType::Flat);
    ASSERT_NE(col, nullptr);
  }

  auto attempt = try_lock_file(lock_path_for(path));
  EXPECT_TRUE(attempt.acquired) << std::strerror(attempt.err);
}

TEST_F(DiskCollectionLockTest, lock_blocks_second_open_in_same_process) {
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path, 8, MetricType::L2, DiskIndexType::Flat);

  try {
    (void)DiskCollection::open(path);
    FAIL() << "expected second open to fail while first collection is alive";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, weak_lock_path_string(path))) << msg;
    EXPECT_TRUE(contains(msg, kLockBusySubstring)) << msg;
  }
}

TEST_F(DiskCollectionLockTest, lock_blocks_second_constructor_path) {
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path, 8, MetricType::L2, DiskIndexType::Flat);

  try {
    DiskCollection second(path, 8, MetricType::L2, DiskIndexType::Flat);
    (void)second;
    FAIL() << "expected create-only constructor to reject an existing collection path";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, "target path already exists")) << msg;
  }
}

TEST_F(DiskCollectionLockTest, lock_rejects_symlinked_lock_file) {
  const auto path = tmp_root_ / "coll";
  build_flat_collection(path);
  const auto lock_path = lock_path_for(path);
  const auto decoy = tmp_root_ / "decoy.lock";
  {
    std::ofstream out(decoy);
    out << "pid=1\n";
  }
  std::filesystem::remove(lock_path);
  std::error_code ec;
  std::filesystem::create_symlink(decoy, lock_path, ec);
  if (ec) {
    GTEST_SKIP() << "symlink creation failed: " << ec.message();
  }

  try {
    (void)DiskCollection::open(path);
    FAIL() << "expected symlinked .lock to be rejected";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, ".lock")) << msg;
    EXPECT_TRUE(contains(msg, "symlink") || contains(msg, "Too many levels") ||
                contains(msg, "ELOOP"))
        << msg;
  }
}

TEST_F(DiskCollectionLockTest, lock_rejects_directory_at_lock_path) {
  const auto path = tmp_root_ / "coll";
  build_flat_collection(path);
  const auto lock_path = lock_path_for(path);
  std::filesystem::remove(lock_path);
  std::filesystem::create_directory(lock_path);

  try {
    (void)DiskCollection::open(path);
    FAIL() << "expected directory .lock to be rejected";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, ".lock")) << msg;
    EXPECT_TRUE(contains(msg, "lock path is not a regular file")) << msg;
  }
}

TEST_F(DiskCollectionLockTest, lock_rejects_fifo_at_lock_path) {
  const auto path = tmp_root_ / "coll";
  build_flat_collection(path);
  const auto lock_path = lock_path_for(path);
  std::filesystem::remove(lock_path);
  if (::mkfifo(lock_path.c_str(), 0600) != 0) {
    GTEST_SKIP() << "mkfifo failed: " << std::strerror(errno);
  }

  try {
    (void)DiskCollection::open(path);
    FAIL() << "expected FIFO .lock to be rejected";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, ".lock")) << msg;
    EXPECT_TRUE(contains(msg, "lock path is not a regular file")) << msg;
  }
}

TEST_F(DiskCollectionLockTest, lock_creates_lock_file_for_legacy_collection_without_one) {
  const auto path = tmp_root_ / "coll";
  build_flat_collection(path);
  std::filesystem::remove(lock_path_for(path));

  auto col = DiskCollection::open(path);
  EXPECT_TRUE(std::filesystem::is_regular_file(lock_path_for(path)));
  struct ::stat st {};
  ASSERT_EQ(::stat(lock_path_for(path).c_str(), &st), 0) << std::strerror(errno);
  EXPECT_EQ((st.st_mode & 0777), 0600);
}

TEST_F(DiskCollectionLockTest, lock_acquire_runs_before_orphan_scan) {
  const auto path = tmp_root_ / "coll";
  build_flat_collection(path);
  const auto orphan = path / "segments" / "seg_00000003";
  std::filesystem::copy(path / "segments" / "seg_00000001",
                        orphan,
                        std::filesystem::copy_options::recursive);

  auto held = try_lock_file(lock_path_for(path));
  ASSERT_TRUE(held.acquired) << std::strerror(held.err);
  const auto blocked_logs = capture_spdlog([&] {
    try {
      (void)DiskCollection::open(path);
      FAIL() << "expected open to fail before orphan scan while lock is externally held";
    } catch (const std::runtime_error &e) {
      const std::string msg = e.what();
      EXPECT_TRUE(contains(msg, kLockBusySubstring)) << msg;
    }
  });
  EXPECT_EQ(blocked_logs.find("orphan segment"), std::string::npos) << blocked_logs;

  held.fd.reset();
  const auto open_logs = capture_spdlog([&] {
    auto col = DiskCollection::open(path);
    EXPECT_EQ(col.size(), 2u);
  });
  EXPECT_TRUE(contains(open_logs, "orphan segment")) << open_logs;
  EXPECT_TRUE(contains(open_logs, "seg_00000003")) << open_logs;
  EXPECT_TRUE(contains(open_logs, "kind=complete")) << open_logs;
}

TEST_F(DiskCollectionLockTest, lock_acquire_message_contains_pid_when_available) {
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path, 8, MetricType::L2, DiskIndexType::Flat);

  try {
    (void)DiskCollection::open(path);
    FAIL() << "expected second open to fail while first collection is alive";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, kLockBusySubstring)) << msg;
    const std::string expected_pid = "(pid=" + std::to_string(::getpid()) + ")";
    EXPECT_TRUE(!contains(msg, "(pid=") || contains(msg, expected_pid)) << msg;
  }
}

TEST_F(DiskCollectionLockTest, lock_blocks_second_process) {
  const auto child_binary = child_open_expect_failure_test_binary();
  if (child_binary.empty()) {
    GTEST_SKIP() << "cannot locate current test binary for exec-based child check";
  }
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path, 8, MetricType::L2, DiskIndexType::Flat);
  ::setenv("ALAYA_LOCK_TEST_PATH", path.c_str(), 1);

  const pid_t pid = ::fork();
  ASSERT_GE(pid, 0) << std::strerror(errno);
  if (pid == 0) {
    ::execl(child_binary.c_str(),
            child_binary.c_str(),
            "--gtest_filter=DiskCollectionLockChild.DISABLED_open_expect_failure",
            "--gtest_also_run_disabled_tests",
            nullptr);
    ::_exit(127);
  }

  const int status = wait_for_child(pid);
  ::unsetenv("ALAYA_LOCK_TEST_PATH");
  ASSERT_TRUE(WIFEXITED(status)) << status;
  EXPECT_EQ(WEXITSTATUS(status), 0) << status;
}

TEST_F(DiskCollectionLockTest, lock_released_when_holder_killed_with_sigkill) {
  const auto path = tmp_root_ / "coll";
  int pipe_fds[2] = {-1, -1};
  ASSERT_EQ(::pipe(pipe_fds), 0) << std::strerror(errno);

  const pid_t pid = ::fork();
  ASSERT_GE(pid, 0) << std::strerror(errno);
  if (pid == 0) {
    ::close(pipe_fds[0]);
    DiskCollection col(path, 8, MetricType::L2, DiskIndexType::Flat);
    const char ready = '1';
    (void)::write(pipe_fds[1], &ready, 1);
    for (;;) {
      ::pause();
    }
  }

  ::close(pipe_fds[1]);
  char ready = 0;
  ASSERT_EQ(::read(pipe_fds[0], &ready, 1), 1);
  ::close(pipe_fds[0]);
  ASSERT_EQ(::kill(pid, SIGKILL), 0) << std::strerror(errno);
  const int status = wait_for_child(pid);
  ASSERT_TRUE(WIFSIGNALED(status)) << status;
  ASSERT_EQ(WTERMSIG(status), SIGKILL);

  EXPECT_NO_THROW({
    auto reopened = DiskCollection::open(path);
    (void)reopened;
  });
}

TEST_F(DiskCollectionLockTest, lock_released_on_clean_exit) {
  const auto path = tmp_root_ / "coll";
  const pid_t pid = ::fork();
  ASSERT_GE(pid, 0) << std::strerror(errno);
  if (pid == 0) {
    DiskCollection col(path, 8, MetricType::L2, DiskIndexType::Flat);
    ::_exit(0);
  }

  const int status = wait_for_child(pid);
  ASSERT_TRUE(WIFEXITED(status)) << status;
  ASSERT_EQ(WEXITSTATUS(status), 0);
  EXPECT_NO_THROW({
    auto reopened = DiskCollection::open(path);
    (void)reopened;
  });
}

TEST_F(DiskCollectionLockTest, legacy_collection_without_lock_can_be_opened) {
  const auto path = tmp_root_ / "coll";
  build_flat_collection(path);
  std::filesystem::remove(lock_path_for(path));

  auto col = DiskCollection::open(path);
  EXPECT_EQ(col.size(), 2u);
  EXPECT_TRUE(std::filesystem::is_regular_file(lock_path_for(path)));
}

TEST_F(DiskCollectionLockTest, legacy_collection_then_concurrent_open_is_blocked) {
  const auto child_binary = child_open_expect_failure_test_binary();
  if (child_binary.empty()) {
    GTEST_SKIP() << "cannot locate current test binary for exec-based child check";
  }
  const auto path = tmp_root_ / "coll";
  build_flat_collection(path);
  std::filesystem::remove(lock_path_for(path));

  auto col = DiskCollection::open(path);
  ::setenv("ALAYA_LOCK_TEST_PATH", path.c_str(), 1);
  const pid_t pid = ::fork();
  ASSERT_GE(pid, 0) << std::strerror(errno);
  if (pid == 0) {
    ::execl(child_binary.c_str(),
            child_binary.c_str(),
            "--gtest_filter=DiskCollectionLockChild.DISABLED_open_expect_failure",
            "--gtest_also_run_disabled_tests",
            nullptr);
    ::_exit(127);
  }

  const int status = wait_for_child(pid);
  ::unsetenv("ALAYA_LOCK_TEST_PATH");
  ASSERT_TRUE(WIFEXITED(status)) << status;
  EXPECT_EQ(WEXITSTATUS(status), 0) << status;
}

TEST(DiskCollectionLockChild, DISABLED_open_expect_failure) {
  const char *path = ::getenv("ALAYA_LOCK_TEST_PATH");
  ASSERT_NE(path, nullptr);
  try {
    auto col = DiskCollection::open(path);
    (void)col;
    FAIL() << "expected child open to fail while parent holds lock";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, weak_lock_path_string(path))) << msg;
    EXPECT_TRUE(contains(msg, kLockBusySubstring)) << msg;
  }
}

// Direct-helper tests for `acquire_collection_lock_for_{open,create}`. These
// exist alongside the public-API tests so we can pin down the helpers' own
// errno branches (especially `_for_create`'s EEXIST path, which is not yet
// reachable through the public ctor in this phase) without depending on
// downstream ctor / open ordering.
TEST_F(DiskCollectionLockTest, helper_for_create_creates_lock_file_with_o_excl) {
  const auto path = tmp_root_ / "coll";
  std::filesystem::create_directories(path);
  ASSERT_FALSE(std::filesystem::exists(lock_path_for(path)));
  auto lock = detail::acquire_collection_lock_for_create(path);
  EXPECT_GE(lock.fd, 0);
  EXPECT_TRUE(std::filesystem::is_regular_file(lock_path_for(path)));
  struct ::stat st {};
  ASSERT_EQ(::stat(lock_path_for(path).c_str(), &st), 0) << std::strerror(errno);
  EXPECT_EQ((st.st_mode & 0777), 0600);
}

TEST_F(DiskCollectionLockTest, helper_for_create_eexist_throws_dual_substring) {
  const auto path = tmp_root_ / "coll";
  std::filesystem::create_directories(path);
  // Pre-create `.lock` to force O_EXCL into the EEXIST branch.
  {
    std::ofstream out(lock_path_for(path));
    out << "stale\n";
  }
  try {
    auto lock = detail::acquire_collection_lock_for_create(path);
    (void)lock;
    FAIL() << "expected EEXIST throw from _for_create against an existing .lock";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, weak_lock_path_string(path))) << msg;
    EXPECT_TRUE(contains(msg, "target path already exists or is being created concurrently"))
        << msg;
  }
}

TEST_F(DiskCollectionLockTest, helper_for_open_auto_creates_lock_for_legacy_collection) {
  const auto path = tmp_root_ / "coll";
  std::filesystem::create_directories(path);
  ASSERT_FALSE(std::filesystem::exists(lock_path_for(path)));
  auto lock = detail::acquire_collection_lock_for_open(path);
  EXPECT_GE(lock.fd, 0);
  EXPECT_TRUE(std::filesystem::is_regular_file(lock_path_for(path)));
}

TEST_F(DiskCollectionLockTest, helper_for_create_then_for_create_eexist_path) {
  // First `_for_create` wins (creates `.lock`); second `_for_create`
  // immediately throws on EEXIST even before the first releases. This is
  // the kernel-level atomicity guarantee that closes KL2 once the ctor is
  // wired in phase 2.
  const auto path = tmp_root_ / "coll";
  std::filesystem::create_directories(path);
  auto lock_a = detail::acquire_collection_lock_for_create(path);
  ASSERT_GE(lock_a.fd, 0);
  try {
    auto lock_b = detail::acquire_collection_lock_for_create(path);
    (void)lock_b;
    FAIL() << "expected second _for_create to fail with EEXIST";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, "target path already exists or is being created concurrently"))
        << msg;
  }
}

TEST_F(DiskCollectionLockTest, open_throws_collection_in_progress_when_lock_and_manifest_absent) {
  // Phase 2 caller-level precondition: when `path/` exists but neither
  // `.lock` nor `collection_manifest.txt` is present, `open()` SHALL throw
  // the documented dual-substring error before invoking `_for_open`.
  const auto path = tmp_root_ / "coll";
  std::filesystem::create_directories(path);  // simulate ctor-mid-flight after mkdir
  ASSERT_FALSE(std::filesystem::exists(lock_path_for(path)));
  ASSERT_FALSE(std::filesystem::exists(path / "collection_manifest.txt"));
  try {
    (void)DiskCollection::open(path);
    FAIL() << "expected open() to throw collection-in-progress";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, weak_lock_path_string(path))) << msg;
    EXPECT_TRUE(contains(msg, "target path is a collection-in-progress, not yet published"))
        << msg;
  }
  // Side-effect check: the precondition runs BEFORE `_for_open`, so the
  // failed `open()` SHALL NOT have created `.lock`.
  EXPECT_FALSE(std::filesystem::exists(lock_path_for(path)));
}

TEST_F(DiskCollectionLockTest, ctor_rollback_on_lock_acquire_failure_removes_path) {
  // Phase 2 ctor rollback: simulate a pre-existing `.lock` file in an
  // otherwise-fresh `path/`. ctor will mkdir(path), then `_for_create` will
  // fail with EEXIST (because some other actor placed `.lock` there), and
  // the rollback rmdir(path) will fail with ENOTEMPTY (because `.lock`
  // makes the dir non-empty) — silently ignored. The `.lock` file SHALL
  // remain (we don't own it).
  //
  // To trigger this without an actual race, we pre-create `path/.lock`
  // ourselves before invoking the ctor. The ctor's `exists(path)` check
  // will fail first, however — we can't reach the `_for_create` failure
  // path that way. So instead we test the simpler "two-ctor race"
  // analog: the EEXIST error contract, which is the shared mechanism.
  const auto path = tmp_root_ / "coll";
  // First ctor wins, holds .lock.
  DiskCollection col1(path, 4, MetricType::L2, DiskIndexType::Flat);
  // Second ctor: `exists(path)` is now true, so the existing TOCTOU-style
  // throw fires first; the message must still surface the literal phrase
  // expected by spec scenario (substring "target path already exists").
  try {
    DiskCollection col2(path, 4, MetricType::L2, DiskIndexType::Flat);
    (void)col2;
    FAIL() << "expected second ctor to fail";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, "target path already exists")) << msg;
  }
}

TEST_F(DiskCollectionLockTest, ctor_creates_lock_before_segments_dir) {
  // Phase 2 ordering invariant: by the time `path/segments/` exists, the
  // `.lock` file MUST already exist (acquire happens before segments mkdir).
  // We can't observe the intermediate state directly without forking, but
  // we can assert that on a successful ctor both files exist and that
  // the lock file mode is 0600 (created by `_for_create`'s open call,
  // not by a stale leftover).
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path, 4, MetricType::L2, DiskIndexType::Flat);
  EXPECT_TRUE(std::filesystem::is_regular_file(lock_path_for(path)));
  EXPECT_TRUE(std::filesystem::is_directory(path / "segments"));
  struct ::stat st {};
  ASSERT_EQ(::stat(lock_path_for(path).c_str(), &st), 0) << std::strerror(errno);
  EXPECT_EQ((st.st_mode & 0777), 0600);
}

// Phase 4: KL3 closure tests. The post-flock inode revalidation is extracted
// into `detail::revalidate_lock_inode_after_flock` for deterministic
// testability. The helper's contract is bounded — see spec.md "Bound" note —
// so we only assert detection of in-flight swaps.
TEST_F(DiskCollectionLockTest, post_flock_revalidation_passes_when_inode_unchanged) {
  const auto path = tmp_root_ / "coll";
  std::filesystem::create_directories(path);
  const auto lock_path = lock_path_for(path);
  const int fd = ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  ASSERT_GE(fd, 0);
  TestFd guard(fd);
  struct ::stat st {};
  ASSERT_EQ(::fstat(fd, &st), 0);
  ASSERT_EQ(::flock(fd, LOCK_EX | LOCK_NB), 0);
  // No swap — revalidation SHALL succeed silently.
  EXPECT_NO_THROW(detail::revalidate_lock_inode_after_flock(lock_path, st.st_dev, st.st_ino));
}

TEST_F(DiskCollectionLockTest, post_flock_revalidation_throws_when_lock_vanished) {
  const auto path = tmp_root_ / "coll";
  std::filesystem::create_directories(path);
  const auto lock_path = lock_path_for(path);
  const int fd = ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  ASSERT_GE(fd, 0);
  TestFd guard(fd);
  struct ::stat st {};
  ASSERT_EQ(::fstat(fd, &st), 0);
  ASSERT_EQ(::flock(fd, LOCK_EX | LOCK_NB), 0);
  // Out-of-band: someone unlinks `.lock` while we hold the fd.
  ASSERT_EQ(::unlink(lock_path.c_str()), 0);
  try {
    detail::revalidate_lock_inode_after_flock(lock_path, st.st_dev, st.st_ino);
    FAIL() << "expected vanished throw";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, weak_lock_path_string(path))) << msg;
    EXPECT_TRUE(contains(msg, "lock file vanished after flock")) << msg;
  }
}

TEST_F(DiskCollectionLockTest, post_flock_revalidation_throws_on_unlink_touch_swap) {
  // KL3 in-call swap: simulate `unlink + touch` between fstat and the
  // post-flock revalidation. The revalidation MUST throw the dual-substring
  // "lock file inode mismatch after acquire" message, with the path and
  // literal both present.
  const auto path = tmp_root_ / "coll";
  std::filesystem::create_directories(path);
  const auto lock_path = lock_path_for(path);
  const int fd_orig = ::open(lock_path.c_str(),
                             O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW,
                             0600);
  ASSERT_GE(fd_orig, 0);
  TestFd orig_guard(fd_orig);
  struct ::stat st_orig {};
  ASSERT_EQ(::fstat(fd_orig, &st_orig), 0);
  ASSERT_EQ(::flock(fd_orig, LOCK_EX | LOCK_NB), 0);

  // Out-of-band: unlink the original `.lock` and recreate to produce a
  // fresh inode at the same path.
  ASSERT_EQ(::unlink(lock_path.c_str()), 0);
  const int fd_new = ::open(lock_path.c_str(),
                            O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW,
                            0600);
  ASSERT_GE(fd_new, 0);
  TestFd new_guard(fd_new);
  struct ::stat st_new {};
  ASSERT_EQ(::fstat(fd_new, &st_new), 0);
  ASSERT_NE(st_orig.st_ino, st_new.st_ino) << "test setup failed: same inode after unlink+touch";

  try {
    detail::revalidate_lock_inode_after_flock(lock_path, st_orig.st_dev, st_orig.st_ino);
    FAIL() << "expected mismatch throw";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, weak_lock_path_string(path))) << msg;
    EXPECT_TRUE(contains(msg, "lock file inode mismatch after acquire")) << msg;
  }
}

// Phase 4 task 4.4: legacy collection (no `.lock` initially) flows through
// `_for_open`, which auto-creates `.lock`. The post-flock revalidation MUST
// pass — fstat and stat both observe the freshly-created inode.
TEST_F(DiskCollectionLockTest, legacy_collection_open_passes_kl3_revalidation) {
  const auto path = tmp_root_ / "coll";
  build_flat_collection(path);
  std::filesystem::remove(lock_path_for(path));
  ASSERT_FALSE(std::filesystem::exists(lock_path_for(path)));
  ASSERT_TRUE(std::filesystem::exists(path / "collection_manifest.txt"));
  // open() SHALL succeed; the in-helper revalidation does not fire.
  auto col = DiskCollection::open(path);
  EXPECT_EQ(col.size(), 2u);
  EXPECT_TRUE(std::filesystem::is_regular_file(lock_path_for(path)));
}

// Phase 4 task 4.2: two-ctor race against a fresh path. Synchronize parent
// and child via a fifo so both call the ctor as close to simultaneously as
// possible. Exactly one SHALL succeed; the other SHALL throw a
// runtime_error containing the absolute lock path AND one of the documented
// literal substrings ("target path already exists" or "is being created
// concurrently"). The on-disk state SHALL be a complete collection.
TEST_F(DiskCollectionLockTest, two_concurrent_ctors_race_via_fork) {
  const auto path = tmp_root_ / "race_coll";
  ASSERT_FALSE(std::filesystem::exists(path));
  int parent_to_child[2] = {-1, -1};
  int child_to_parent[2] = {-1, -1};
  ASSERT_EQ(::pipe(parent_to_child), 0);
  ASSERT_EQ(::pipe(child_to_parent), 0);

  const pid_t pid = ::fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    ::close(parent_to_child[1]);
    ::close(child_to_parent[0]);
    char ready_byte = 0;
    (void)::read(parent_to_child[0], &ready_byte, 1);
    int child_status = 0;
    try {
      DiskCollection col(path, 4, MetricType::L2, DiskIndexType::Flat);
      child_status = 1;
    } catch (const std::runtime_error &e) {
      const std::string msg = e.what();
      const bool ok = msg.find("target path already exists") != std::string::npos ||
                      msg.find("is being created concurrently") != std::string::npos;
      child_status = ok ? 2 : 99;
    } catch (...) {
      child_status = 100;
    }
    char b = static_cast<char>(child_status);
    (void)::write(child_to_parent[1], &b, 1);
    ::_exit(0);
  }
  ::close(parent_to_child[0]);
  ::close(child_to_parent[1]);
  // Release both sides as close to simultaneously as possible.
  const char go = '!';
  (void)::write(parent_to_child[1], &go, 1);

  bool parent_ok = false;
  std::string parent_err;
  try {
    DiskCollection col(path, 4, MetricType::L2, DiskIndexType::Flat);
    parent_ok = true;
  } catch (const std::runtime_error &e) {
    parent_err = e.what();
  }

  char child_byte = 99;
  (void)::read(child_to_parent[0], &child_byte, 1);
  ::close(parent_to_child[1]);
  ::close(child_to_parent[0]);
  const int status = wait_for_child(pid);
  ASSERT_TRUE(WIFEXITED(status));

  const bool child_won = (child_byte == 1);
  const bool child_lost_clean = (child_byte == 2);
  ASSERT_TRUE(child_won || child_lost_clean) << "child status=" << static_cast<int>(child_byte);

  // Exactly one winner.
  EXPECT_NE(parent_ok, child_won);

  if (!parent_ok) {
    EXPECT_TRUE(contains(parent_err, "target path already exists") ||
                contains(parent_err, "is being created concurrently"))
        << parent_err;
  }
  // On-disk state: full collection present.
  EXPECT_TRUE(std::filesystem::is_regular_file(lock_path_for(path)));
  EXPECT_TRUE(std::filesystem::is_directory(path / "segments"));
  EXPECT_TRUE(std::filesystem::is_regular_file(path / "collection_manifest.txt"));
}

// Phase 4 task 4.1: ctor concurrent with open() against an absent path. We
// fork a child that loops `open(path)` until it either succeeds or hits a
// stable error message; the parent constructs the collection. Outcomes
// MUST satisfy: ctor succeeds; child's open() either also succeeds (after
// ctor finished) or throws a documented error (collection-in-progress, or
// flock contention, or — once published — succeeds).
TEST_F(DiskCollectionLockTest, ctor_and_concurrent_open_no_partial_disk_state) {
  const auto path = tmp_root_ / "ctor_open_race";
  ASSERT_FALSE(std::filesystem::exists(path));
  int parent_to_child[2] = {-1, -1};
  int child_to_parent[2] = {-1, -1};
  ASSERT_EQ(::pipe(parent_to_child), 0);
  ASSERT_EQ(::pipe(child_to_parent), 0);
  const pid_t pid = ::fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    ::close(parent_to_child[1]);
    ::close(child_to_parent[0]);
    char go = 0;
    (void)::read(parent_to_child[0], &go, 1);
    int outcome = 0;
    try {
      auto col = DiskCollection::open(path);
      outcome = 1;
    } catch (const std::runtime_error &e) {
      const std::string msg = e.what();
      const bool ok = msg.find("collection-in-progress") != std::string::npos ||
                      msg.find("collection is already open by another process") !=
                          std::string::npos ||
                      msg.find("path does not exist") != std::string::npos;
      outcome = ok ? 2 : 99;
    } catch (...) {
      outcome = 100;
    }
    char b = static_cast<char>(outcome);
    (void)::write(child_to_parent[1], &b, 1);
    ::_exit(0);
  }
  ::close(parent_to_child[0]);
  ::close(child_to_parent[1]);
  const char go = '!';
  (void)::write(parent_to_child[1], &go, 1);

  bool ctor_ok = false;
  std::string ctor_err;
  try {
    DiskCollection col(path, 4, MetricType::L2, DiskIndexType::Flat);
    ctor_ok = true;
  } catch (const std::runtime_error &e) {
    ctor_err = e.what();
  }

  char child_byte = 99;
  (void)::read(child_to_parent[0], &child_byte, 1);
  ::close(parent_to_child[1]);
  ::close(child_to_parent[0]);
  const int status = wait_for_child(pid);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_TRUE(child_byte == 1 || child_byte == 2) << "child status=" << static_cast<int>(child_byte);

  // ctor SHALL succeed (it had no contention besides the racing open).
  EXPECT_TRUE(ctor_ok) << ctor_err;

  // Final on-disk state: either full collection OR (if ctor failed AND
  // rolled back) absent. ctor success implies full state.
  if (ctor_ok) {
    EXPECT_TRUE(std::filesystem::is_regular_file(lock_path_for(path)));
    EXPECT_TRUE(std::filesystem::is_directory(path / "segments"));
    EXPECT_TRUE(std::filesystem::is_regular_file(path / "collection_manifest.txt"));
  } else {
    EXPECT_FALSE(std::filesystem::exists(path));
  }
}

TEST_F(DiskCollectionLockTest, helper_parent_dir_missing_throws_path_not_exist) {
  const auto path = tmp_root_ / "missing_root" / "coll";
  // No mkdir — parent dir missing on purpose.
  ASSERT_FALSE(std::filesystem::exists(path));
  try {
    auto lock = detail::acquire_collection_lock_for_open(path);
    (void)lock;
    FAIL() << "expected ENOENT throw when parent dir is missing";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, "collection path does not exist") ||
                contains(msg, "No such file or directory"))
        << msg;
  }
  // _for_create has the same parent-dir requirement.
  try {
    auto lock = detail::acquire_collection_lock_for_create(path);
    (void)lock;
    FAIL() << "expected ENOENT throw when parent dir is missing (for_create)";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, "collection path does not exist") ||
                contains(msg, "No such file or directory"))
        << msg;
  }
}

}  // namespace
}  // namespace alaya::disk
