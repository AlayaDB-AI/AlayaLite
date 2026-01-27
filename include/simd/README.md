# SIMD Distance Functions

This module provides optimized SIMD implementations for distance calculations used in vector similarity search.

## Table of Contents

- [Overview](#overview)
- [L2 Distance (Euclidean)](#l2-distance-euclidean)
  - [Full Precision (FP32)](#l2-full-precision-fp32)
  - [SQ8 Quantized](#l2-sq8-quantized)
  - [SQ4 Quantized](#l2-sq4-quantized)
- [Inner Product Distance](#inner-product-distance)
  - [Full Precision (FP32)](#ip-full-precision-fp32)
  - [SQ8 Quantized](#ip-sq8-quantized)
  - [SQ4 Quantized](#ip-sq4-quantized)

---

## Overview

This module supports multiple SIMD instruction sets:

| Instruction Set | Description |
|-----------------|-------------|
| **Generic** | Baseline implementation (compiler auto-optimizes to SSE2) |
| **AVX2** | 256-bit SIMD, widely supported on modern CPUs |
| **AVX-512** | 512-bit SIMD, available on newer Intel/AMD CPUs |
| **AUTO** | Automatic dispatch to the best available implementation |

> **Note**: Benchmark results show that AVX2 often outperforms AVX-512 for certain operations due to reduced clock throttling and better cache utilization.

---

## L2 Distance (Euclidean)

### L2 Full Precision (FP32)

**Recommended**: AVX2 provides the best performance for L2 distance calculations.

| Dimension | Generic | AVX2 | AVX-512 | AUTO |
|:---------:|--------:|-----:|--------:|-----:|
| 96 | 13.32 ns | **10.05 ns (1.33x)** | 12.84 ns (1.04x) | **10.23 ns (1.30x)** |
| 128 | 11.49 ns | 11.30 ns (1.02x) | 12.29 ns (0.93x) | 11.31 ns (1.02x) |
| 256 | 22.44 ns | **20.00 ns (1.12x)** | 23.59 ns (0.95x) | **20.12 ns (1.12x)** |
| 384 | 34.18 ns | **31.18 ns (1.10x)** | 35.47 ns (0.96x) | **31.24 ns (1.09x)** |
| 512 | 45.47 ns | **39.63 ns (1.15x)** | 47.22 ns (0.96x) | **39.91 ns (1.14x)** |
| 768 | 67.98 ns | **54.25 ns (1.25x)** | 70.09 ns (0.97x) | **54.60 ns (1.25x)** |
| 960 | 86.06 ns | **66.34 ns (1.30x)** | 86.62 ns (0.99x) | **66.17 ns (1.30x)** |
| 1024 | 90.73 ns | **69.55 ns (1.30x)** | 91.89 ns (0.99x) | **69.48 ns (1.31x)** |
| 1536 | 140.76 ns | **102.08 ns (1.38x)** | 150.45 ns (0.94x) | **102.25 ns (1.38x)** |

<details>
<summary>Benchmark Details</summary>

- **Function**: `get_l2_sqr_func()` with auto dispatch
- **SIMD Level**: AVX-512 capable CPU
- **Iterations**: 1,000,000 per test
- **Bold** indicates >5% speedup over Generic baseline

</details>

---

### L2 SQ8 Quantized

**Recommended**: AVX-512 provides the best performance for SQ8 quantized data.

| Dimension | Generic | AVX2 | AVX-512 | AUTO |
|:---------:|--------:|-----:|--------:|-----:|
| 96 | 28.30 ns | 29.11 ns (0.97x) | **17.96 ns (1.58x)** | **17.89 ns (1.58x)** |
| 128 | 27.87 ns | 36.24 ns (0.77x) | **21.96 ns (1.27x)** | **22.20 ns (1.26x)** |
| 256 | 51.92 ns | 61.96 ns (0.84x) | **40.32 ns (1.29x)** | **40.90 ns (1.27x)** |
| 384 | 72.81 ns | 88.19 ns (0.83x) | **58.19 ns (1.25x)** | **58.20 ns (1.25x)** |
| 512 | 93.32 ns | 113.66 ns (0.82x) | **73.26 ns (1.27x)** | **73.30 ns (1.27x)** |
| 768 | 134.78 ns | 165.07 ns (0.82x) | **113.31 ns (1.19x)** | **113.54 ns (1.19x)** |
| 960 | 164.83 ns | 194.15 ns (0.85x) | **127.01 ns (1.30x)** | **127.82 ns (1.30x)** |
| 1024 | 174.73 ns | 216.46 ns (0.81x) | **134.36 ns (1.30x)** | **134.65 ns (1.30x)** |
| 1536 | 258.92 ns | 320.31 ns (0.81x) | **222.06 ns (1.17x)** | **221.21 ns (1.17x)** |

<details>
<summary>Benchmark Details</summary>

- **Function**: `get_l2_sqr_sq8_func()` with auto dispatch
- **SIMD Level**: AVX-512 capable CPU
- **Iterations**: 1,000,000 per test
- **Bold** indicates >5% speedup over Generic baseline

</details>

---

### L2 SQ4 Quantized

**Recommended**: AVX2 provides exceptional performance for SQ4 quantized data (up to 6x speedup).

> SQ4 packs 2 values per byte (4 bits each), providing significant memory savings.

| Dimension | Generic | AVX2 | AVX-512 | AUTO |
|:---------:|--------:|-----:|--------:|-----:|
| 96 | 165.73 ns | **32.34 ns (5.13x)** | 155.66 ns (1.06x) | **32.68 ns (5.07x)** |
| 128 | 209.39 ns | **42.18 ns (4.96x)** | 208.26 ns (1.01x) | **41.56 ns (5.04x)** |
| 256 | 410.58 ns | **73.95 ns (5.55x)** | 384.05 ns (1.07x) | **73.90 ns (5.56x)** |
| 384 | 614.86 ns | **106.03 ns (5.80x)** | 574.73 ns (1.07x) | **106.00 ns (5.80x)** |
| 512 | 831.23 ns | **139.13 ns (5.97x)** | 771.59 ns (1.08x) | **138.91 ns (5.98x)** |
| 768 | 1237.25 ns | **202.98 ns (6.10x)** | 1141.28 ns (1.08x) | **202.19 ns (6.12x)** |
| 960 | 1551.63 ns | **253.75 ns (6.11x)** | 1436.89 ns (1.08x) | **249.74 ns (6.21x)** |
| 1024 | 1647.45 ns | **263.13 ns (6.26x)** | 1539.20 ns (1.07x) | **266.75 ns (6.18x)** |
| 1536 | 2467.69 ns | **395.55 ns (6.24x)** | 2289.45 ns (1.08x) | **399.39 ns (6.18x)** |

<details>
<summary>Benchmark Details</summary>

- **Function**: `get_l2_sqr_sq4_func()` with auto dispatch
- **SIMD Level**: AVX-512 capable CPU
- **Iterations**: 100,000 per test
- **Bold** indicates >5% speedup over Generic baseline

</details>

---

## Inner Product Distance

### IP Full Precision (FP32)

**Recommended**: AVX2 provides the best performance for inner product calculations.

| Dimension | Generic | AVX2 | AVX-512 | AUTO |
|:---------:|--------:|-----:|--------:|-----:|
| 96 | 10.35 ns | **8.91 ns (1.16x)** | 11.02 ns (0.94x) | **8.70 ns (1.19x)** |
| 128 | 17.09 ns | **14.28 ns (1.20x)** | **14.96 ns (1.14x)** | **13.25 ns (1.29x)** |
| 256 | 22.86 ns | **19.55 ns (1.17x)** | 22.53 ns (1.01x) | **19.04 ns (1.20x)** |
| 384 | 35.44 ns | **32.50 ns (1.09x)** | 34.86 ns (1.02x) | **30.80 ns (1.15x)** |
| 512 | 46.29 ns | **39.86 ns (1.16x)** | 46.10 ns (1.00x) | **39.01 ns (1.19x)** |
| 768 | 69.09 ns | **57.98 ns (1.19x)** | 67.52 ns (1.02x) | **57.82 ns (1.19x)** |
| 960 | 84.88 ns | **72.53 ns (1.17x)** | 84.70 ns (1.00x) | **71.98 ns (1.18x)** |
| 1024 | 88.87 ns | **74.60 ns (1.19x)** | 88.76 ns (1.00x) | **74.84 ns (1.19x)** |
| 1536 | 137.30 ns | **111.53 ns (1.23x)** | 134.70 ns (1.02x) | **112.79 ns (1.22x)** |

<details>
<summary>Benchmark Details</summary>

- **Function**: `get_ip_func()` with auto dispatch
- **SIMD Level**: AVX-512 capable CPU
- **Iterations**: 100,000 per test
- **Bold** indicates >5% speedup over Generic baseline

</details>

---

### IP SQ8 Quantized

**Recommended**: AVX-512 provides the best performance for most dimensions.

| Dimension | Generic | AVX2 | AVX-512 | AUTO |
|:---------:|--------:|-----:|--------:|-----:|
| 96 | 33.42 ns | 32.61 ns (1.02x) | **22.33 ns (1.50x)** | **21.67 ns (1.54x)** |
| 128 | 31.08 ns | 35.78 ns (0.87x) | 30.58 ns (1.02x) | 30.65 ns (1.01x) |
| 256 | 54.93 ns | 60.06 ns (0.91x) | **44.17 ns (1.24x)** | **43.94 ns (1.25x)** |
| 384 | 80.31 ns | 83.04 ns (0.97x) | 79.40 ns (1.01x) | 79.67 ns (1.01x) |
| 512 | 106.12 ns | 106.86 ns (0.99x) | 104.90 ns (1.01x) | 105.91 ns (1.00x) |
| 768 | 172.98 ns | **157.09 ns (1.10x)** | 174.15 ns (0.99x) | 174.24 ns (0.99x) |
| 960 | 172.44 ns | 191.59 ns (0.90x) | **153.84 ns (1.12x)** | **153.80 ns (1.12x)** |
| 1024 | 182.98 ns | 203.17 ns (0.90x) | **162.88 ns (1.12x)** | **162.86 ns (1.12x)** |
| 1536 | 345.03 ns | **302.35 ns (1.14x)** | 341.95 ns (1.01x) | 342.38 ns (1.01x) |

<details>
<summary>Benchmark Details</summary>

- **Function**: `get_ip_sq8_func()` with auto dispatch
- **SIMD Level**: AVX-512 capable CPU
- **Iterations**: 100,000 per test
- **Bold** indicates >5% speedup over Generic baseline

</details>

---

### IP SQ4 Quantized

**Recommended**: AVX2 provides exceptional performance for SQ4 quantized data (up to 6x speedup).

> SQ4 packs 2 values per byte (4 bits each), providing significant memory savings.

| Dimension | Generic | AVX2 | AVX-512 | AUTO |
|:---------:|--------:|-----:|--------:|-----:|
| 96 | 173.15 ns | **34.15 ns (5.07x)** | 154.71 ns (1.12x) | **34.21 ns (5.06x)** |
| 128 | 205.35 ns | **43.01 ns (4.77x)** | 206.04 ns (1.00x) | **43.85 ns (4.68x)** |
| 256 | 409.87 ns | **74.80 ns (5.48x)** | 384.12 ns (1.07x) | **74.85 ns (5.48x)** |
| 384 | 614.65 ns | **106.32 ns (5.78x)** | 571.19 ns (1.08x) | **106.53 ns (5.77x)** |
| 512 | 829.00 ns | **139.69 ns (5.93x)** | 768.06 ns (1.08x) | **139.61 ns (5.94x)** |
| 768 | 1237.62 ns | **201.34 ns (6.15x)** | 1141.16 ns (1.08x) | **201.50 ns (6.14x)** |
| 960 | 1544.20 ns | **248.72 ns (6.21x)** | 1418.06 ns (1.09x) | **248.89 ns (6.20x)** |
| 1024 | 1647.53 ns | **264.93 ns (6.22x)** | 1511.38 ns (1.09x) | **265.09 ns (6.21x)** |
| 1536 | 2464.98 ns | **391.47 ns (6.30x)** | 2247.32 ns (1.10x) | **391.27 ns (6.30x)** |

<details>
<summary>Benchmark Details</summary>

- **Function**: `get_ip_sq4_func()` with auto dispatch
- **SIMD Level**: AVX-512 capable CPU
- **Iterations**: 100,000 per test
- **Bold** indicates >5% speedup over Generic baseline

</details>

---

## Performance Summary

| Distance Type | Data Type | Best Implementation | Typical Speedup |
|--------------|-----------|---------------------|-----------------|
| L2 | FP32 | AVX2 | 1.1x - 1.4x |
| L2 | SQ8 | AVX-512 | 1.2x - 1.6x |
| L2 | SQ4 | AVX2 | 5x - 6x |
| IP | FP32 | AVX2 | 1.1x - 1.3x |
| IP | SQ8 | AVX-512 | 1.1x - 1.5x |
| IP | SQ4 | AVX2 | 5x - 6x |

> **Key Insight**: SQ4 quantization with AVX2 provides the most dramatic performance improvement, achieving up to 6x speedup while reducing memory usage by 8x compared to FP32.
