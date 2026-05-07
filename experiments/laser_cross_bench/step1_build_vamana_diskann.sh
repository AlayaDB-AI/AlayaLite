#!/usr/bin/env bash
# Build DiskANN Vamana graph for GIST-1M.
# Uses numactl --interleave=all + 48 threads to match bench NUMA policy.
# Output: /md1/huangliang/alaya-dev/tmp/laser_cross_bench/vamana_diskann/gist_vamana.index
set -euo pipefail

DISKANN=/md1/huangliang/alaya-dev/Laser/DiskANN/build/apps/build_memory_index
DATA=/md1/huangliang/alaya-dev/data/gist1m/gist_base.fbin
OUT_DIR=/md1/huangliang/alaya-dev/tmp/laser_cross_bench/vamana_diskann
PREFIX="${OUT_DIR}/gist_vamana.index"

mkdir -p "$OUT_DIR"
echo "[diskann-vamana] building R=64 L=200 alpha=1.2 threads=48 → ${PREFIX}"

numactl --interleave=all \
    "$DISKANN" \
    --data_type float \
    --dist_fn l2 \
    --data_path "$DATA" \
    --index_path_prefix "$PREFIX" \
    -R 64 \
    -L 200 \
    --alpha 1.2 \
    --num_threads 48

echo "[diskann-vamana] done → ${PREFIX}"
ls -lh "${OUT_DIR}/"
