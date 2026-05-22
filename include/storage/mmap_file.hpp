// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

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

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#else
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
    HANDLE hFile = ::CreateFileW(path.c_str(),
                                 GENERIC_READ,
                                 FILE_SHARE_READ,
                                 nullptr,
                                 OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL,
                                 nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("MMapFile open failed: " + path.string() + ": Win32 error " +
                               std::to_string(::GetLastError()));
    }
    // Read-only mmap rejects directories and non-regular files implicitly:
    // CreateFileMapping on a directory handle fails with ERROR_INVALID_HANDLE.
    LARGE_INTEGER li{};
    if (!::GetFileSizeEx(hFile, &li)) {
      const auto err = ::GetLastError();
      ::CloseHandle(hFile);
      throw std::runtime_error("MMapFile size query failed: " + path.string() + ": Win32 error " +
                               std::to_string(err));
    }
    if (li.QuadPart <= 0) {
      ::CloseHandle(hFile);
      throw std::runtime_error("MMapFile rejects empty file: " + path.string());
    }
    HANDLE hMap = ::CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (hMap == nullptr) {
      const auto err = ::GetLastError();
      ::CloseHandle(hFile);
      throw std::runtime_error("MMapFile CreateFileMapping failed: " + path.string() +
                               ": Win32 error " + std::to_string(err));
    }
    void *p = ::MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (p == nullptr) {
      const auto err = ::GetLastError();
      ::CloseHandle(hMap);
      ::CloseHandle(hFile);
      throw std::runtime_error("MMapFile MapViewOfFile failed: " + path.string() +
                               ": Win32 error " + std::to_string(err));
    }
    // The view holds a reference on the section; the section holds one on the
    // file. We can close both handles now and the mapping survives until
    // UnmapViewOfFile.
    ::CloseHandle(hMap);
    ::CloseHandle(hFile);
    data_ = p;
    size_ = static_cast<size_t>(li.QuadPart);
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
#ifdef _WIN32
    if (data_ != nullptr) {
      ::UnmapViewOfFile(data_);
    }
#else
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
