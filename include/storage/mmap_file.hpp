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

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#ifndef _WIN32
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace alaya::storage {

static_assert(sizeof(size_t) >= 8, "MMapFile v1 requires a 64-bit host (size_t >= 8 bytes)");

namespace detail {

// Overload pair for both strerror_r ABIs:
//   - POSIX (XSI): int strerror_r(int, char *, size_t);  → 0 on success.
//   - GNU:         char *strerror_r(int, char *, size_t); → may NOT write to buf;
//                  returned pointer is the message.
// We pass the strerror_r return value plus the scratch buffer here and the
// compiler picks the matching overload. Both overloads are always compiled,
// but neither references strerror_r itself, so each is valid on any host.
inline auto strerror_r_to_string(int rc, char *buf) -> std::string {
  return rc == 0 ? std::string(buf) : std::string{};
}

inline auto strerror_r_to_string(char *result, char * /*buf*/) -> std::string {
  return result != nullptr ? std::string(result) : std::string{};
}

inline auto errno_to_string(int err) -> std::string {
#ifndef _WIN32
  char buf[256] = {0};
  auto str = strerror_r_to_string(::strerror_r(err, buf, sizeof(buf)), buf);
  if (str.empty()) {
    return "errno=" + std::to_string(err);
  }
  return str;
#else
  return "errno=" + std::to_string(err);
#endif
}

}  // namespace detail

class MMapFile {
 public:
  MMapFile() = default;

  explicit MMapFile(const std::filesystem::path &path) { open_impl(path); }

  static auto open(const std::filesystem::path &path) -> MMapFile { return MMapFile(path); }

  ~MMapFile() { release(); }

  MMapFile(const MMapFile &) = delete;
  auto operator=(const MMapFile &) -> MMapFile & = delete;

  MMapFile(MMapFile &&other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  auto operator=(MMapFile &&other) noexcept -> MMapFile & {
    if (this != &other) {
      release();
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  auto data() const -> const void * { return data_; }
  auto size() const -> size_t { return size_; }

  template <typename T>
  auto as() const -> const T * {
    if (size_ % sizeof(T) != 0) {
      throw std::runtime_error("MMapFile::as<T>: file size " + std::to_string(size_) +
                               " is not a multiple of sizeof(T)=" + std::to_string(sizeof(T)));
    }
    return static_cast<const T *>(data_);
  }

 private:
  void open_impl(const std::filesystem::path &path) {
#ifdef _WIN32
    (void)path;
    throw std::runtime_error("MMapFile: unsupported on this platform in v1");
#else
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
      int saved = errno;
      throw std::runtime_error("MMapFile open failed: " + path.string() + ": " +
                               detail::errno_to_string(saved));
    }
    struct ::stat st{};
    if (::fstat(fd, &st) != 0) {
      int saved = errno;
      ::close(fd);
      throw std::runtime_error("MMapFile fstat failed: " + path.string() + ": " +
                               detail::errno_to_string(saved));
    }
    if (!S_ISREG(st.st_mode)) {
      ::close(fd);
      throw std::runtime_error("MMapFile rejects non-regular file: " + path.string());
    }
    if (st.st_size <= 0) {
      ::close(fd);
      throw std::runtime_error("MMapFile rejects empty file: " + path.string());
    }
    auto sz = static_cast<size_t>(st.st_size);
    void *p = ::mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    int mmap_err = errno;
    ::close(fd);
    if (p == MAP_FAILED) {
      throw std::runtime_error("MMapFile mmap failed: " + path.string() + ": " +
                               detail::errno_to_string(mmap_err));
    }
    data_ = p;
    size_ = sz;
#endif
  }

  void release() noexcept {
#ifndef _WIN32
    if (data_ != nullptr) {
      ::munmap(data_, size_);
    }
#endif
    data_ = nullptr;
    size_ = 0;
  }

  void *data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace alaya::storage
