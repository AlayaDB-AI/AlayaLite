"""Generate synth_20k_768d dataset for Laser alignment Tier A daily gate.

Produces:
    data/synth_20k_768d/base.fbin     # 20000 x 768 float32, DiskANN fbin header
    data/synth_20k_768d/query.fbin    # 1000  x 768 float32
    data/synth_20k_768d/gt.ibin       # 1000  x 100 int32 top-100 L2 neighbours
    data/synth_20k_768d/README.md     # seed, parameters, sha256 of all three files

Design (see openspec/changes/align-laser-with-upstream/design.md D5):
- 10-cluster Gaussian mixture, centers drawn on unit sphere (seed=42).
- Each base / query point: center_c + N(0, sigma^2 I_768), sigma=0.3.
- All points rotated by a fixed random-orthogonal Q (seed=42).
- Ground truth is top-100 by L2 over the rotated base (brute force, float32 BLAS).
- main_dim=256, degree=64 → node_len_=6144 > kSectorLen=4096 → node_per_page_=1,
  which avoids the preserved qg_builder.hpp:256 page-layout bug.

Usage:
    python scripts/laser_alignment/gen_synth_20k_768d.py \\
        --out /md1/huangliang/alaya-dev/data/synth_20k_768d
"""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import numpy as np

SEED: int = 42
NUM_CLUSTERS: int = 10
DIM: int = 768
N_BASE: int = 20_000
N_QUERY: int = 1_000
SIGMA: float = 0.3
TOP_K: int = 100


def _write_fbin(path: Path, data: np.ndarray) -> None:
    assert data.dtype == np.float32
    n, d = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, d))
        data.tofile(f)


def _write_ibin(path: Path, data: np.ndarray) -> None:
    assert data.dtype == np.int32
    n, k = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, k))
        data.tofile(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gen_cluster_centers(rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal(size=(NUM_CLUSTERS, DIM)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw


def _gen_orthogonal_q(rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal(size=(DIM, DIM))
    q, r = np.linalg.qr(a)
    d = np.sign(np.diag(r))
    d[d == 0] = 1.0
    q = q * d
    return q.astype(np.float32)


def _gen_points(rng: np.random.Generator, n: int, centers: np.ndarray) -> np.ndarray:
    labels = rng.integers(0, NUM_CLUSTERS, size=n)
    noise = rng.standard_normal(size=(n, DIM)).astype(np.float32) * np.float32(SIGMA)
    return centers[labels] + noise


def _brute_force_top_k(base: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    base_sq = (base * base).sum(axis=1, dtype=np.float32)
    gt = np.empty((query.shape[0], k), dtype=np.int32)
    chunk_q = 64
    for start in range(0, query.shape[0], chunk_q):
        end = min(start + chunk_q, query.shape[0])
        q_block = query[start:end]
        dots = q_block @ base.T
        d2 = base_sq[None, :] - 2.0 * dots
        part = np.argpartition(d2, k, axis=1)[:, :k]
        rows = np.arange(part.shape[0])[:, None]
        part_d2 = d2[rows, part]
        order = np.argsort(part_d2, axis=1)
        gt[start:end] = part[rows, order].astype(np.int32)
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    print(f"[gen] seed={SEED}, N_base={N_BASE}, N_query={N_QUERY}, dim={DIM}, sigma={SIGMA}")
    centers = _gen_cluster_centers(rng)
    q_rot = _gen_orthogonal_q(rng)

    print("[gen] sampling base points …")
    base = _gen_points(rng, N_BASE, centers)

    print("[gen] sampling query points …")
    query = _gen_points(rng, N_QUERY, centers)

    print("[gen] applying fixed orthogonal rotation Q …")
    base_rot = (base @ q_rot).astype(np.float32, copy=False)
    query_rot = (query @ q_rot).astype(np.float32, copy=False)

    base_path = out / "base.fbin"
    query_path = out / "query.fbin"
    gt_path = out / "gt.ibin"

    print(f"[gen] writing {base_path}")
    _write_fbin(base_path, base_rot)
    print(f"[gen] writing {query_path}")
    _write_fbin(query_path, query_rot)

    print("[gen] computing top-100 brute-force ground truth …")
    gt = _brute_force_top_k(base_rot, query_rot, TOP_K)
    print(f"[gen] writing {gt_path}")
    _write_ibin(gt_path, gt)

    print("[gen] hashing …")
    sha_base = _sha256(base_path)
    sha_query = _sha256(query_path)
    sha_gt = _sha256(gt_path)

    readme = out / "README.md"
    readme.write_text(
        "# synth_20k_768d\n"
        "\n"
        "Synthetic ANN dataset for Laser alignment Tier A daily gate.\n"
        "See `openspec/changes/align-laser-with-upstream/design.md` D5 for rationale.\n"
        "\n"
        "## Parameters\n"
        f"- seed = {SEED}\n"
        f"- num_clusters = {NUM_CLUSTERS} (centers unit-normalised on S^{DIM - 1})\n"
        f"- dim = {DIM}\n"
        f"- N_base = {N_BASE}\n"
        f"- N_query = {N_QUERY}\n"
        f"- sigma = {SIGMA} (Gaussian mixture stddev, isotropic)\n"
        f"- rotation = random orthogonal Q (drawn from seed={SEED})\n"
        f"- gt metric = L2, top-{TOP_K}, brute force float32\n"
        "\n"
        "## Layout guarantee\n"
        "At `main_dim=256, degree=64`:\n"
        "`node_len_ = (32*256 + 32*512 + 128*64 + 64*256)/8 = 6144` bytes\n"
        "> `kSectorLen = 4096` → `node_per_page_ = 1`, which sidesteps the\n"
        "preserved `qg_builder.hpp:256` page-layout bug.\n"
        "\n"
        "## Reproduction\n"
        "```\n"
        "python scripts/laser_alignment/gen_synth_20k_768d.py --out <this dir>\n"
        "```\n"
        "\n"
        "## sha256\n"
        f"- base.fbin  {sha_base}\n"
        f"- query.fbin {sha_query}\n"
        f"- gt.ibin    {sha_gt}\n"
    )

    print("[gen] done.")
    print(f"  base.fbin  sha256 {sha_base}")
    print(f"  query.fbin sha256 {sha_query}")
    print(f"  gt.ibin    sha256 {sha_gt}")


if __name__ == "__main__":
    main()
