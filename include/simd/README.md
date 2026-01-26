# distance_l2.hpp

Because the local tests show that using avx2 results in better effects, we have chosen avx2 as the best option in auto.
## L2 SIMD Distance Performance Comparison

| Dimension | Generic (baseline) | AVX2 | AVX-512 | AUTO |
|-----------|-------------------|------|---------|------|
| 96 | 13.32 ns (1.00x) | **10.05 ns (1.33x)** | 12.84 ns (1.04x) | **10.23 ns (1.30x)** |
| 128 | 11.49 ns (1.00x) | 11.30 ns (1.02x) | 12.29 ns (0.93x) | 11.31 ns (1.02x) |
| 256 | 22.44 ns (1.00x) | **20.00 ns (1.12x)** | 23.59 ns (0.95x) | **20.12 ns (1.12x)** |
| 384 | 34.18 ns (1.00x) | **31.18 ns (1.10x)** | 35.47 ns (0.96x) | **31.24 ns (1.09x)** |
| 512 | 45.47 ns (1.00x) | **39.63 ns (1.15x)** | 47.22 ns (0.96x) | **39.91 ns (1.14x)** |
| 768 | 67.98 ns (1.00x) | **54.25 ns (1.25x)** | 70.09 ns (0.97x) | **54.60 ns (1.25x)** |
| 960 | 86.06 ns (1.00x) | **66.34 ns (1.30x)** | 86.62 ns (0.99x) | **66.17 ns (1.30x)** |
| 1024 | 90.73 ns (1.00x) | **69.55 ns (1.30x)** | 91.89 ns (0.99x) | **69.48 ns (1.31x)** |
| 1536 | 140.76 ns (1.00x) | **102.08 ns (1.38x)** | 150.45 ns (0.94x) | **102.25 ns (1.38x)** |

Note: The compiler automatically optimizes to SSE2 for generic

## Summary

- **Bold** indicates >5% speedup over Generic baseline
- **Best** = get_l2_sqr_func() with auto dispatch (uses Fixed for known dims)
- SIMD Level: AVX-512
- Iterations per test: 1000000

# distance_l2_sq8
## SQ8 L2 SIMD Distance Performance Comparison

| Dimension | Generic (baseline) | AVX2 | AVX-512 | AUTO |
|-----------|-------------------|------|---------|------|
| 96 | 28.30 ns (1.00x) | 29.11 ns (0.97x) | **17.96 ns (1.58x)** | **17.89 ns (1.58x)** |
| 128 | 27.87 ns (1.00x) | 36.24 ns (0.77x) | **21.96 ns (1.27x)** | **22.20 ns (1.26x)** |
| 256 | 51.92 ns (1.00x) | 61.96 ns (0.84x) | **40.32 ns (1.29x)** | **40.90 ns (1.27x)** |
| 384 | 72.81 ns (1.00x) | 88.19 ns (0.83x) | **58.19 ns (1.25x)** | **58.20 ns (1.25x)** |
| 512 | 93.32 ns (1.00x) | 113.66 ns (0.82x) | **73.26 ns (1.27x)** | **73.30 ns (1.27x)** |
| 768 | 134.78 ns (1.00x) | 165.07 ns (0.82x) | **113.31 ns (1.19x)** | **113.54 ns (1.19x)** |
| 960 | 164.83 ns (1.00x) | 194.15 ns (0.85x) | **127.01 ns (1.30x)** | **126.82 ns (1.30x)** |
| 1024 | 174.73 ns (1.00x) | 216.46 ns (0.81x) | **134.36 ns (1.30x)** | **134.65 ns (1.30x)** |
| 1536 | 258.92 ns (1.00x) | 320.31 ns (0.81x) | **222.06 ns (1.17x)** | **221.21 ns (1.17x)** |

## Summary

- **Bold** indicates >5% speedup over Generic baseline
- **AUTO** = get_l2_sqr_sq8_func() with auto dispatch
- SIMD Level: AVX-512
- Iterations per test: 1000000

# distance_l2_sq4

## SQ4 L2 SIMD Distance Performance Comparison

| Dimension | Generic (baseline) | AVX2 | AVX-512 | AUTO |
|-----------|-------------------|------|---------|------|
| 96 | 165.73 ns (1.00x) | **32.34 ns (5.13x)** | **155.66 ns (1.06x)** | **32.68 ns (5.07x)** |
| 128 | 209.39 ns (1.00x) | **42.18 ns (4.96x)** | 208.26 ns (1.01x) | **41.56 ns (5.04x)** |
| 256 | 410.58 ns (1.00x) | **73.95 ns (5.55x)** | **384.05 ns (1.07x)** | **73.90 ns (5.56x)** |
| 384 | 614.86 ns (1.00x) | **106.03 ns (5.80x)** | **574.73 ns (1.07x)** | **106.00 ns (5.80x)** |
| 512 | 831.23 ns (1.00x) | **139.13 ns (5.97x)** | **771.59 ns (1.08x)** | **138.91 ns (5.98x)** |
| 768 | 1237.25 ns (1.00x) | **202.98 ns (6.10x)** | **1141.28 ns (1.08x)** | **202.19 ns (6.12x)** |
| 960 | 1551.63 ns (1.00x) | **253.75 ns (6.11x)** | **1436.89 ns (1.08x)** | **249.74 ns (6.21x)** |
| 1024 | 1647.45 ns (1.00x) | **263.13 ns (6.26x)** | **1539.20 ns (1.07x)** | **266.75 ns (6.18x)** |
| 1536 | 2467.69 ns (1.00x) | **395.55 ns (6.24x)** | **2289.45 ns (1.08x)** | **399.39 ns (6.18x)** |

## Summary

- **Bold** indicates >5% speedup over Generic baseline
- **AUTO** = get_l2_sqr_sq4_func() with auto dispatch
- SQ4 packs 2 values per byte (4 bits each)
- SIMD Level: AVX-512
- Iterations per test: 100000
