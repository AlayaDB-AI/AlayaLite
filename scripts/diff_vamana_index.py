#!/usr/bin/env python3
"""
diff_vamana_index.py — structural comparison of two DiskANN-format .index files.

Reads a pair of single-file Vamana `.index` files (the layout written by
AlayaLite's `vamana_writer.hpp` and by DiskANN's `InMemGraphStore::save_graph`)
and reports whether their graphs are structurally similar enough to plausibly
yield equivalent recall.

This is a debugging / investigation tool — not a pass/fail gate. The pass/fail
gate lives in the C++ harness at `tests/index/test_vamana_alignment.cpp`.

Usage:
    python3 diff_vamana_index.py <path_a> <path_b> [--csv <out.csv>] [--bfs-samples N]

Output sections:
    1. Header comparison (file_size, max_observed_degree, start, frozen_pts)
    2. Degree distribution: avg, min, max, and a 10-bin out-degree histogram
    3. In-degree histogram (derived by reversing all edges)
    4. BFS reachability: nodes reached from medoid within K hops, orphan count
    5. Approximate diameter from medoid via sampled BFS
    6. Per-node neighbor-set Jaccard summary
    7. Optional CSV of per-node added/removed neighbor lists

All integers in the .index format are little-endian on x86_64.
"""
from __future__ import annotations

import argparse
import csv
import struct
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass
class VamanaIndex:
    path: Path
    file_size: int
    max_observed_degree: int
    start: int
    frozen_pts: int
    # adjacency[i] = list of neighbor ids of node i
    adjacency: List[List[int]]

    @property
    def num_nodes(self) -> int:
        return len(self.adjacency)


def load_index(path: Path) -> VamanaIndex:
    """Parse the DiskANN single-file .index layout.

    Header (24 bytes):  uint64 expected_file_size, uint32 max_observed_degree,
                        uint32 start, uint64 frozen_pts.
    Body:               for each node id in insertion order (0..N-1),
                        uint32 k, then k × uint32 neighbor ids.
    """
    data = path.read_bytes()
    if len(data) < 24:
        raise ValueError(f"{path}: file shorter than 24-byte header")
    file_size, max_deg, start, frozen = struct.unpack_from("<QIIQ", data, 0)
    if file_size != len(data):
        print(
            f"warn: {path} expected_file_size header={file_size} "
            f"but actual size is {len(data)} (header may be stale)",
            file=sys.stderr,
        )
    adjacency: List[List[int]] = []
    offset = 24
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError(f"{path}: truncated node count at offset {offset}")
        (k,) = struct.unpack_from("<I", data, offset)
        offset += 4
        if offset + 4 * k > len(data):
            raise ValueError(f"{path}: truncated neighbors at node {len(adjacency)}")
        nbrs = list(struct.unpack_from(f"<{k}I", data, offset))
        offset += 4 * k
        adjacency.append(nbrs)
    return VamanaIndex(
        path=path,
        file_size=file_size,
        max_observed_degree=max_deg,
        start=start,
        frozen_pts=frozen,
        adjacency=adjacency,
    )


def _histogram(values: Sequence[int], bins: int) -> List[Tuple[int, int, int]]:
    """Return [(lo, hi, count)] with ``bins`` equal-width integer buckets."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [(lo, hi, len(values))]
    step = (hi - lo + 1) / bins
    result: List[Tuple[int, int, int]] = []
    for b in range(bins):
        bucket_lo = int(lo + b * step)
        bucket_hi = int(lo + (b + 1) * step) - 1
        if b == bins - 1:
            bucket_hi = hi
        count = sum(1 for v in values if bucket_lo <= v <= bucket_hi)
        result.append((bucket_lo, bucket_hi, count))
    return result


def compute_in_degrees(idx: VamanaIndex) -> List[int]:
    in_deg = [0] * idx.num_nodes
    for node_id, nbrs in enumerate(idx.adjacency):
        for m in nbrs:
            if 0 <= m < idx.num_nodes:
                in_deg[m] += 1
    return in_deg


def bfs_levels(idx: VamanaIndex, source: int) -> List[int]:
    """Return the BFS distance array from `source`. -1 means unreachable."""
    depth = [-1] * idx.num_nodes
    depth[source] = 0
    q: deque[int] = deque([source])
    while q:
        n = q.popleft()
        for m in idx.adjacency[n]:
            if 0 <= m < idx.num_nodes and depth[m] == -1:
                depth[m] = depth[n] + 1
                q.append(m)
    return depth


def sampled_eccentricity(idx: VamanaIndex, samples: int, rng_seed: int) -> Tuple[int, int]:
    """Estimate diameter by BFS from `samples` random sources; returns
    (max eccentricity observed, number of unreachable events)."""
    import random

    rng = random.Random(rng_seed)
    n = idx.num_nodes
    if n == 0:
        return (0, 0)
    picks = set()
    picks.add(idx.start)
    while len(picks) < min(samples, n):
        picks.add(rng.randint(0, n - 1))
    max_ecc = 0
    unreachable = 0
    for src in picks:
        depths = bfs_levels(idx, src)
        reachable = [d for d in depths if d >= 0]
        max_ecc = max(max_ecc, max(reachable) if reachable else 0)
        unreachable += sum(1 for d in depths if d < 0)
    return (max_ecc, unreachable)


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def _print_header_cmp(a: VamanaIndex, b: VamanaIndex) -> None:
    print("## Header comparison\n")
    print(f"                              {'A':>20}  {'B':>20}")
    print(f"  path                        {a.path.name:>20}  {b.path.name:>20}")
    print(f"  file_size                   {a.file_size:>20}  {b.file_size:>20}")
    print(f"  max_observed_degree         {a.max_observed_degree:>20}  {b.max_observed_degree:>20}")
    print(f"  start (medoid)              {a.start:>20}  {b.start:>20}")
    print(f"  frozen_pts                  {a.frozen_pts:>20}  {b.frozen_pts:>20}")
    print(f"  num_nodes                   {a.num_nodes:>20}  {b.num_nodes:>20}")
    print()


def _print_degree_stats(idx: VamanaIndex, label: str) -> None:
    out_degs = [len(nb) for nb in idx.adjacency]
    in_degs = compute_in_degrees(idx)
    print(f"## Degree stats ({label})\n")
    print(
        f"  out-degree   avg={sum(out_degs) / max(1, len(out_degs)):.2f}  "
        f"min={min(out_degs)}  max={max(out_degs)}"
    )
    print(
        f"  in-degree    avg={sum(in_degs) / max(1, len(in_degs)):.2f}  "
        f"min={min(in_degs)}  max={max(in_degs)}"
    )
    print("  out-degree histogram (10 bins):")
    for lo, hi, cnt in _histogram(out_degs, 10):
        print(f"    [{lo:>4}, {hi:>4}]  {cnt}")
    print("  in-degree histogram (10 bins):")
    for lo, hi, cnt in _histogram(in_degs, 10):
        print(f"    [{lo:>4}, {hi:>4}]  {cnt}")
    orphans = sum(1 for d in in_degs if d == 0)
    print(f"  orphans (in-degree=0): {orphans}")
    print()


def _print_bfs_stats(idx: VamanaIndex, label: str, bfs_samples: int, seed: int) -> None:
    depths_from_medoid = bfs_levels(idx, idx.start)
    unreachable = sum(1 for d in depths_from_medoid if d < 0)
    reachable = [d for d in depths_from_medoid if d >= 0]
    max_depth = max(reachable) if reachable else 0
    max_ecc, extra_unreachable = sampled_eccentricity(idx, bfs_samples, seed)
    print(f"## BFS reachability ({label})\n")
    print(f"  reachable from medoid     {len(reachable)}/{idx.num_nodes}")
    print(f"  unreachable from medoid   {unreachable}")
    print(f"  max BFS depth from medoid {max_depth}")
    print(
        f"  sampled eccentricity (n={bfs_samples}): "
        f"max_radius={max_ecc}  total_unreachable_events={extra_unreachable}"
    )
    print()


def _print_jaccard(a: VamanaIndex, b: VamanaIndex) -> None:
    if a.num_nodes != b.num_nodes:
        print("## Jaccard\n\n  skipped: different num_nodes")
        return
    n = a.num_nodes
    jaccs = [jaccard(a.adjacency[i], b.adjacency[i]) for i in range(n)]
    mean_j = sum(jaccs) / max(1, n)
    full_match = sum(1 for j in jaccs if j == 1.0)
    no_match = sum(1 for j in jaccs if j == 0.0)
    buckets = [0, 0, 0, 0, 0]
    for j in jaccs:
        idx = min(4, int(j * 5))
        buckets[idx] += 1
    print("## Neighbor-list Jaccard (A vs B, per node)\n")
    print(f"  mean Jaccard              {mean_j:.4f}")
    print(f"  full-match nodes (J=1)    {full_match}")
    print(f"  disjoint nodes   (J=0)    {no_match}")
    print("  Jaccard buckets [0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]:")
    print(f"    {buckets}")
    print()


def _write_csv(out_path: Path, a: VamanaIndex, b: VamanaIndex) -> None:
    n = min(a.num_nodes, b.num_nodes)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "node_id",
                "deg_a",
                "deg_b",
                "jaccard",
                "only_in_a",
                "only_in_b",
                "shared",
            ]
        )
        for i in range(n):
            sa = set(a.adjacency[i])
            sb = set(b.adjacency[i])
            shared = sa & sb
            only_a = sa - sb
            only_b = sb - sa
            j = len(shared) / max(1, len(sa | sb))
            w.writerow(
                [
                    i,
                    len(sa),
                    len(sb),
                    f"{j:.4f}",
                    ";".join(str(x) for x in sorted(only_a)),
                    ";".join(str(x) for x in sorted(only_b)),
                    ";".join(str(x) for x in sorted(shared)),
                ]
            )
    print(f"per-node diff CSV written to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("path_a", type=Path, help="First .index file (e.g. AlayaLite output)")
    p.add_argument("path_b", type=Path, help="Second .index file (e.g. DiskANN reference)")
    p.add_argument("--csv", type=Path, default=None, help="Write per-node diff CSV here")
    p.add_argument(
        "--bfs-samples",
        type=int,
        default=8,
        help="Number of random sources for sampled eccentricity (default 8)",
    )
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    a = load_index(args.path_a)
    b = load_index(args.path_b)

    _print_header_cmp(a, b)
    _print_degree_stats(a, "A")
    _print_degree_stats(b, "B")
    _print_bfs_stats(a, "A", args.bfs_samples, args.seed)
    _print_bfs_stats(b, "B", args.bfs_samples, args.seed)
    _print_jaccard(a, b)

    if args.csv is not None:
        _write_csv(args.csv, a, b)

    return 0


if __name__ == "__main__":
    sys.exit(main())
