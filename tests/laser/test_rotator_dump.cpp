// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Round-trip test for FHTRotator::dump_signs.
//
// Validates that the dump:
//   (a) is byte-identical across two rotators seeded with the same value;
//   (b) differs across two rotators seeded with different values;
//   (c) writes `uint64 paded_dim` + `float32[paded_dim]` in that order, and
//       the header value matches the configured paded_dim.
//
// Note: seed=0 takes the `std::random_device` path; the C++ standard does
// not guarantee that random_device produces distinct values across calls
// on every platform, so we do not assert non-determinism here.
//
// Exit 0 on pass, non-zero + cerr line on the first failure.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "index/graph/laser/utils/rotator.hpp"

namespace {

std::vector<char> read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "cannot open " << path << '\n';
        std::exit(2);
    }
    const auto size = static_cast<std::streamsize>(f.tellg());
    f.seekg(0);
    std::vector<char> buf(static_cast<size_t>(size));
    f.read(buf.data(), size);
    return buf;
}

std::string unique_tmp(const std::string& tag, int pid) {
    namespace fs = std::filesystem;
    return (fs::temp_directory_path() /
            ("alaya_fht_rotator_dump_" + tag + "_" + std::to_string(pid) + ".bin")).string();
}

}  // namespace

int main() {
    using alaya::laser::FHTRotator;

    const size_t dim = 256;  // already a power of two → paded_dim_ == 256
    const uint64_t seed_a = 42;
    const uint64_t seed_b = 43;
    const int pid = static_cast<int>(::getpid());

    // (a) Seeded determinism — two rotators at the same seed.
    FHTRotator r1(dim, seed_a);
    FHTRotator r2(dim, seed_a);

    const std::string path_a = unique_tmp("seeded_a", pid);
    const std::string path_a2 = unique_tmp("seeded_a2", pid);
    r1.dump_signs(path_a);
    r2.dump_signs(path_a2);

    const auto bytes_a = read_all(path_a);
    const auto bytes_a2 = read_all(path_a2);
    if (bytes_a != bytes_a2) {
        std::cerr << "FAIL: same-seed dumps differ (seed=" << seed_a
                  << ", size_a=" << bytes_a.size()
                  << ", size_a2=" << bytes_a2.size() << ")\n";
        return 1;
    }

    // (b) Seed sensitivity — different seeds must give different dumps.
    FHTRotator r3(dim, seed_b);
    const std::string path_b = unique_tmp("seeded_b", pid);
    r3.dump_signs(path_b);
    const auto bytes_b = read_all(path_b);
    if (bytes_a == bytes_b) {
        std::cerr << "FAIL: seeds " << seed_a << " and " << seed_b
                  << " produced identical dumps\n";
        return 1;
    }

    // (c) Header + payload layout.
    const size_t paded_dim = 256;
    const size_t expected_size = sizeof(uint64_t) + sizeof(float) * paded_dim;
    if (bytes_a.size() != expected_size) {
        std::cerr << "FAIL: dump size " << bytes_a.size()
                  << " != expected " << expected_size << '\n';
        return 1;
    }
    uint64_t header = 0;
    std::memcpy(&header, bytes_a.data(), sizeof(uint64_t));
    if (header != paded_dim) {
        std::cerr << "FAIL: header paded_dim " << header
                  << " != expected " << paded_dim << '\n';
        return 1;
    }

    std::filesystem::remove(path_a);
    std::filesystem::remove(path_a2);
    std::filesystem::remove(path_b);

    std::cout << "alaya::laser::FHTRotator::dump_signs round-trip OK "
              << "(paded_dim=" << paded_dim << ", bytes=" << bytes_a.size() << ")\n";
    return 0;
}
