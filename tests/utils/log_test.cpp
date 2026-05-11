// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "utils/log.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <spdlog/sinks/ostream_sink.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include "utils/openmp.hpp"
#include "utils/platform.hpp"
#include "utils/platform_fs.hpp"

namespace alaya {

namespace {

class ScopedDefaultLogger {
 public:
  explicit ScopedDefaultLogger(std::ostream &stream)
      : original_logger_(spdlog::default_logger()), original_level_(spdlog::get_level()) {
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(stream);
    auto logger = std::make_shared<spdlog::logger>("log-test", std::move(sink));
    logger->set_pattern("%v");
    logger->set_level(spdlog::level::trace);
    spdlog::set_default_logger(std::move(logger));
    spdlog::set_level(spdlog::level::trace);
  }

  ~ScopedDefaultLogger() {
    spdlog::set_default_logger(original_logger_);
    spdlog::set_level(original_level_);
  }

 private:
  std::shared_ptr<spdlog::logger> original_logger_;
  spdlog::level::level_enum original_level_;
};

auto unique_temp_dir() -> std::filesystem::path {
  auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("alayalite_platform_fs_test_" + std::to_string(suffix));
}

auto read_text_file(const std::filesystem::path &path) -> std::string {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

auto count_substrings(const std::string &text, const std::string &needle) -> size_t {
  size_t count = 0;
  size_t pos = text.find(needle);
  while (pos != std::string::npos) {
    ++count;
    pos = text.find(needle, pos + needle.size());
  }
  return count;
}

auto normalize_path_separators(std::string text) -> std::string {
  std::replace(text.begin(), text.end(), '\\', '/');
  return text;
}

ALAYA_NOINLINE auto noinline_identity(int value) -> int { return value; }

ALAYA_ALWAYS_INLINE auto always_inline_identity(int value) -> int { return value; }

FAST_BEGIN
ALAYA_NOINLINE auto fast_section_add(int lhs, int rhs) -> int { return lhs + rhs; }
FAST_END

}  // namespace

TEST(LogTest, Show) {
  LOG_TRACE("test {}", "tracing");
  LOG_DEBUG("test {}", "debug");
  LOG_INFO("test {}", "info");
  LOG_WARN("test {}", "warn");
  LOG_ERROR("test {}", "error");
  LOG_CRITICAL("test {}", "critical");

  EXPECT_EQ(0, 0);
}

TEST(LogTest, ExtractRelativePathHandlesSourceMarkersAndFallbacks) {
  EXPECT_STREQ(extract_relative_path("/repo/include/utils/log.hpp"), "include/utils/log.hpp");
  EXPECT_STREQ(extract_relative_path("C:\\repo\\include\\utils\\log.hpp"),
               "include\\utils\\log.hpp");
  EXPECT_STREQ(extract_relative_path("/repo/src/module.cpp"), "src/module.cpp");
  EXPECT_STREQ(extract_relative_path("C:\\repo\\src\\module.cpp"), "src\\module.cpp");
  EXPECT_STREQ(extract_relative_path("/repo/tests/utils/log_test.cpp"), "tests/utils/log_test.cpp");
  EXPECT_STREQ(extract_relative_path("C:\\repo\\tests\\utils\\log_test.cpp"),
               "tests\\utils\\log_test.cpp");
  EXPECT_STREQ(extract_relative_path("/tmp/generated.cpp"), "generated.cpp");
  EXPECT_STREQ(extract_relative_path("C:\\tmp\\generated.cpp"), "generated.cpp");
  EXPECT_STREQ(extract_relative_path("generated.cpp"), "generated.cpp");
}

TEST(LogTest, MacrosEmitFormattedMessagesAndLogOnce) {
  std::ostringstream stream;
  ScopedDefaultLogger logger_guard(stream);

  LOG_TRACE("trace {}", 1);
  LOG_DEBUG("debug {}", 2);
  LOG_INFO("info {}", 3);
  LOG_WARN("warn {}", 4);
  LOG_ERROR("error {}", 5);
  LOG_CRITICAL("critical {}", 6);
  for (int i = 0; i < 3; ++i) {
    LOG_INFO_ONCE("only once {}", i);
  }

  spdlog::default_logger()->flush();
  const auto output = normalize_path_separators(stream.str());

  EXPECT_NE(output.find("[Alaya] [tests/utils/log_test.cpp:"), std::string::npos);
  EXPECT_NE(output.find("trace 1"), std::string::npos);
  EXPECT_NE(output.find("debug 2"), std::string::npos);
  EXPECT_NE(output.find("info 3"), std::string::npos);
  EXPECT_NE(output.find("warn 4"), std::string::npos);
  EXPECT_NE(output.find("error 5"), std::string::npos);
  EXPECT_NE(output.find("critical 6"), std::string::npos);
  EXPECT_EQ(count_substrings(output, "only once"), 1U);
}

TEST(PlatformTest, AllocationHelpersHandleEdgeCasesAndAlignment) {
  EXPECT_EQ(alaya_aligned_alloc_impl(0, 64), nullptr);
  EXPECT_EQ(alaya_aligned_alloc_impl(64, 0), nullptr);

  void *ptr = alaya_aligned_alloc_impl(65, 64);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64U, 0U);
  alaya_aligned_free_impl(ptr);
  alaya_aligned_free_impl(nullptr);
}

TEST(PlatformTest, CompilerHelperMacrosAreUsable) {
  EXPECT_EQ(ALAYA_LIKELY(true), true);
  EXPECT_EQ(ALAYA_UNLIKELY(false), false);
  EXPECT_EQ(noinline_identity(7), 7);
  EXPECT_EQ(always_inline_identity(8), 8);
  EXPECT_EQ(fast_section_add(2, 3), 5);

#if defined(__linux__)
  EXPECT_TRUE(true);
#elif defined(_WIN32)
  EXPECT_TRUE(true);
#elif defined(__APPLE__)
  EXPECT_TRUE(true);
#else
  EXPECT_TRUE(true);
#endif
}

TEST(OpenMPTest, HelpersAreCallableOnCurrentPlatform) {
  EXPECT_EQ(platform::openmp_enabled(),
#ifdef _OPENMP
            true
#else
            false
#endif
  );
  platform::set_openmp_thread_count(2);
  EXPECT_GE(platform::openmp_thread_num(), 0);
}

TEST(PlatformFsTest, CreateAndSyncHelpersHandleEmptyMissingAndExistingPaths) {
  namespace fs = std::filesystem;

  const auto root = unique_temp_dir();
  const auto nested = root / "nested";
  const auto file = nested / "data.txt";

  platform::create_directories_if_needed({});
  platform::create_directories_if_needed(nested);
  ASSERT_TRUE(fs::is_directory(nested));

  platform::sync_file({});
  platform::sync_file(root / "missing.txt");
  platform::sync_directory({});
  platform::sync_directory(root / "missing_dir");

  {
    std::ofstream output(file, std::ios::binary);
    output << "payload";
  }

  platform::sync_file(file);
  platform::sync_directory(nested);

  std::error_code ec;
  fs::remove_all(root, ec);
}

TEST(PlatformFsTest, AtomicReplaceCreatesParentsReplacesFilesAndReportsFailures) {
  namespace fs = std::filesystem;

  const auto root = unique_temp_dir();
  const auto source = root / "source.txt";
  const auto target = root / "nested" / "target.txt";

  fs::create_directories(root);
  {
    std::ofstream output(source, std::ios::binary);
    output << "first";
  }

  platform::atomic_replace(source, target);
  EXPECT_FALSE(fs::exists(source));
  ASSERT_TRUE(fs::exists(target));
  EXPECT_EQ(read_text_file(target), "first");

  {
    std::ofstream output(source, std::ios::binary);
    output << "second";
  }

  platform::atomic_replace(source, target);
  EXPECT_EQ(read_text_file(target), "second");

  EXPECT_THROW(platform::atomic_replace(root / "missing.txt", target), std::runtime_error);
  EXPECT_TRUE(fs::exists(target));
  EXPECT_EQ(read_text_file(target), "second");

  std::error_code ec;
  fs::remove_all(root, ec);
}
}  // namespace alaya
