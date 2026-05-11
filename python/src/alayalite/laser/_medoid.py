# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Medoid generation for graph-based nearest neighbor search entry points."""

import faiss
import numpy as np

from alayalite.laser._io import read_fbin, write_fbin, write_ibin


def generate_medoids(base_path, n_clusters, sample_ratio=0.10, seed=None):
    """
    Generate medoid vectors by K-means clustering on base vectors.

    Medoids are actual data points closest to cluster centroids,
    used as entry points for graph-based search.

    Args:
        base_path: Path to PCA-transformed base vectors (fbin format)
        n_clusters: Number of clusters (entry points) to generate
        sample_ratio: Fraction of data to sample for clustering
        seed: RNG seed. ``None`` preserves upstream Laser's non-deterministic
            behaviour (unseeded ``np.random.choice`` + faiss kmeans with
            time-based seed). Pass an int to make medoid selection
            reproducible — required for any alignment or byte-equality gate
            across runs.

    Returns:
        Tuple of (medoid_indices, medoid_vectors) as numpy arrays
    """
    X = read_fbin(base_path)  # pylint: disable=invalid-name
    n, d = X.shape
    sample_size = int(sample_ratio * n)
    print(f"Total vectors: {n}, dimension: {d}, sample size for clustering: {sample_size}")

    if n > sample_size:
        rng = np.random.default_rng(seed) if seed is not None else np.random
        sample_indices = np.sort(rng.choice(n, sample_size, replace=False))
        # Sorted fancy indexing on the mmap touches pages roughly sequentially —
        # avoids the per-vector seek+read syscall that the previous open-loop did.
        sample_vectors = np.ascontiguousarray(X[sample_indices], dtype=np.float32)
    else:
        sample_vectors = X
        sample_indices = np.arange(n)

    index = faiss.index_factory(d, f"IVF{n_clusters},Flat")
    index.verbose = True
    if seed is not None:
        # faiss kmeans defaults to a time-based seed on every train(); fix it
        # so the cluster-center selection is reproducible.
        index.cp.seed = int(seed)
    index.train(sample_vectors)
    index.add(sample_vectors)

    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    index.nprobe = 5
    _, nearest = index.search(centroids, 1)
    medoid_local_indices = nearest.flatten()

    medoid_indices = sample_indices[medoid_local_indices].reshape(-1, 1)
    medoid_vectors = sample_vectors[medoid_local_indices]

    return medoid_indices, medoid_vectors


def generate_and_save_medoids(base_path, indices_path, vectors_path, n_clusters, seed=None):
    """
    Generate medoids and save to binary files.

    Args:
        base_path: Path to PCA-transformed base vectors
        indices_path: Output path for medoid indices (ibin)
        vectors_path: Output path for medoid vectors (fbin)
        n_clusters: Number of clusters
        seed: RNG seed threaded through to ``generate_medoids``.
    """
    medoid_indices, medoid_vectors = generate_medoids(base_path, n_clusters, seed=seed)
    write_ibin(indices_path, medoid_indices)
    write_fbin(vectors_path, medoid_vectors)
