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

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "storage/mmap_file.hpp"

#ifndef _WIN32
#include <unistd.h>
#endif

namespace alaya::storage {

namespace {

class MMapFileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto pid_str = std::to_string(static_cast<long long>(::getpid()));
    auto base = std::filesystem::temp_directory_path() /
                ("alaya_mmap_file_test_" + pid_str + "_" +
                 ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(base);
    tmp_dir_ = base;
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir_, ec);
  }

  auto write_bytes(const std::string &name, const void *data, size_t len) -> std::filesystem::path {
    auto p = tmp_dir_ / name;
    std::ofstream ofs(p, std::ios::binary | std::ios::trunc);
    if (len > 0) {
      ofs.write(static_cast<const char *>(data), static_cast<std::streamsize>(len));
    }
    ofs.close();
    return p;
  }

  std::filesystem::path tmp_dir_;
};

#ifndef _WIN32

TEST_F(MMapFileTest, BasicOpenRead) {
  const std::array<uint8_t, 4> bytes{0x00, 0x01, 0x02, 0x03};
  auto p = write_bytes("basic.bin", bytes.data(), bytes.size());

  MMapFile m(p);
  ASSERT_NE(m.data(), nullptr);
  EXPECT_EQ(m.size(), bytes.size());
  const auto *got = static_cast<const uint8_t *>(m.data());
  for (size_t i = 0; i < bytes.size(); ++i) {
    EXPECT_EQ(got[i], bytes[i]) << "byte mismatch at offset " << i;
  }
}

TEST_F(MMapFileTest, MissingFileThrowsWithErrnoText) {
  auto missing = tmp_dir_ / "definitely_does_not_exist.bin";
  try {
    (void)MMapFile(missing);
    FAIL() << "expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("open"), std::string::npos)
        << "message must reference 'open': " << msg;
    auto enoent_text = detail::errno_to_string(ENOENT);
    if (!enoent_text.empty() && enoent_text.rfind("errno=", 0) != 0) {
      EXPECT_NE(msg.find(enoent_text), std::string::npos)
          << "message must contain ENOENT description '" << enoent_text << "', got: " << msg;
    }
  }
}

TEST_F(MMapFileTest, EmptyFileThrows) {
  auto p = write_bytes("empty.bin", nullptr, 0);
  ASSERT_TRUE(std::filesystem::exists(p));
  ASSERT_EQ(std::filesystem::file_size(p), 0u);
  try {
    (void)MMapFile(p);
    FAIL() << "expected exception on empty file";
  } catch (const std::exception &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("empty"), std::string::npos)
        << "message must contain 'empty': " << msg;
  }
}

TEST_F(MMapFileTest, MoveTransfersOwnership) {
  const std::array<uint8_t, 8> bytes{0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04};
  auto p = write_bytes("move.bin", bytes.data(), bytes.size());

  MMapFile a(p);
  const void *original_ptr = a.data();
  const size_t original_size = a.size();

  MMapFile b = std::move(a);
  EXPECT_EQ(b.data(), original_ptr);
  EXPECT_EQ(b.size(), original_size);
  EXPECT_EQ(a.data(), nullptr);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(a.size(), 0u);       // NOLINT(bugprone-use-after-move)

  // Move-assignment also works and leaves the source empty.
  MMapFile c;
  c = std::move(b);
  EXPECT_EQ(c.data(), original_ptr);
  EXPECT_EQ(c.size(), original_size);
  EXPECT_EQ(b.data(), nullptr);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(b.size(), 0u);       // NOLINT(bugprone-use-after-move)
}

TEST_F(MMapFileTest, AsTMisalignedThrows) {
  const std::array<uint8_t, 7> seven{1, 2, 3, 4, 5, 6, 7};
  auto p = write_bytes("seven.bin", seven.data(), seven.size());
  MMapFile m(p);
  ASSERT_EQ(m.size(), 7u);
  try {
    (void)m.as<int32_t>();
    FAIL() << "expected exception on size%sizeof(T) != 0";
  } catch (const std::exception &e) {
    const std::string msg = e.what();
    const bool mentions_size = msg.find("size") != std::string::npos ||
                               msg.find("sizeof") != std::string::npos;
    EXPECT_TRUE(mentions_size) << "message must reference 'size' or 'sizeof', got: " << msg;
  }
}

TEST_F(MMapFileTest, AsTAlignedSucceeds) {
  // Sanity: size that IS a multiple of sizeof(int32_t) does not throw.
  const std::array<int32_t, 3> ints{0x11111111, 0x22222222, 0x33333333};
  auto p = write_bytes("ints.bin", ints.data(), sizeof(ints));
  MMapFile m(p);
  const auto *got = m.as<int32_t>();
  ASSERT_NE(got, nullptr);
  EXPECT_EQ(got[0], 0x11111111);
  EXPECT_EQ(got[1], 0x22222222);
  EXPECT_EQ(got[2], 0x33333333);
}

TEST_F(MMapFileTest, NoFdLeakAfterOpen) {
  const std::array<uint8_t, 16> bytes{};
  auto p = write_bytes("nfd.bin", bytes.data(), bytes.size());

  auto count_open_fds = []() -> size_t {
    std::error_code ec;
    size_t n = 0;
    for (const auto &entry : std::filesystem::directory_iterator("/proc/self/fd", ec)) {
      (void)entry;
      ++n;
    }
    return n;
  };

  const size_t before = count_open_fds();
  ASSERT_GT(before, 0u) << "/proc/self/fd should be readable";

  constexpr size_t kN = 1024;
  std::vector<MMapFile> mappings;
  mappings.reserve(kN);
  for (size_t i = 0; i < kN; ++i) {
    mappings.emplace_back(p);
  }

  const size_t after = count_open_fds();
  // Allow a small slack — gtest internals, /proc iteration fd, etc. The point
  // is "does not grow by ~1024", not "is exactly equal".
  EXPECT_LT(after, before + 32u) << "fd count grew too much: before=" << before
                                  << " after=" << after << " for kN=" << kN;

  // Each mapping is still readable through data() after the fd was closed.
  for (size_t i = 0; i < kN; ++i) {
    ASSERT_NE(mappings[i].data(), nullptr) << "mapping " << i << " lost its data ptr";
    ASSERT_EQ(mappings[i].size(), bytes.size());
  }
}

TEST_F(MMapFileTest, SymlinkRejected) {
  const std::array<uint8_t, 4> bytes{1, 2, 3, 4};
  auto target = write_bytes("real.bin", bytes.data(), bytes.size());
  auto link = tmp_dir_ / "link.bin";
  std::error_code ec;
  std::filesystem::create_symlink(target, link, ec);
  if (ec) {
    GTEST_SKIP() << "symlink creation failed (filesystem may not support it): " << ec.message();
  }
  try {
    (void)MMapFile(link);
    FAIL() << "expected runtime_error on symlink path";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    auto eloop_text = detail::errno_to_string(ELOOP);
    if (!eloop_text.empty() && eloop_text.rfind("errno=", 0) != 0) {
      EXPECT_NE(msg.find(eloop_text), std::string::npos)
          << "message must contain ELOOP text '" << eloop_text << "', got: " << msg;
    }
  }
}

#else  // _WIN32

TEST_F(MMapFileTest, WindowsUnsupportedThrows) {
  const std::array<uint8_t, 4> bytes{1, 2, 3, 4};
  auto p = write_bytes("win.bin", bytes.data(), bytes.size());
  try {
    (void)MMapFile(p);
    FAIL() << "expected runtime_error on Windows";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("unsupported on this platform in v1"), std::string::npos)
        << "message must contain 'unsupported on this platform in v1', got: " << msg;
  }
}

#endif

TEST_F(MMapFileTest, DefaultConstructedIsEmpty) {
  MMapFile m;
  EXPECT_EQ(m.data(), nullptr);
  EXPECT_EQ(m.size(), 0u);
}

}  // namespace
}  // namespace alaya::storage
