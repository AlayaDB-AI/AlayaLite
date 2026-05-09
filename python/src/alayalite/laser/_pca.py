# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PCA transformation utilities for vector datasets."""

import os
import struct

import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from alayalite.laser._io import read_fbin


def save_pca_params(pca, filepath):
    """
    Save PCA parameters to binary file for C++ loading.

    File format:
      - uint64: dimension
      - float32[dim]: mean vector
      - float32[dim * dim]: components matrix (row-major)
    """
    dim = pca.n_components_
    mean = pca.mean_.astype(np.float32)
    components = pca.components_.astype(np.float32)

    with open(filepath, "wb") as f:
        f.write(struct.pack("<Q", dim))
        mean.tofile(f)
        components.tofile(f)

    print(f"PCA parameters saved to {filepath}")
    print(f"  - Dimension: {dim}")
    print(f"  - Mean shape: {mean.shape}")
    print(f"  - Components shape: {components.shape}")


def _canonicalize_pca_sign(pca):
    """Pin PCA component signs so two sklearn versions produce identical output.

    sklearn's ``svd_flip`` picks per-row signs from the max-magnitude
    element of U/VT, and that choice changed between 1.7.x and 1.8.x for
    near-zero max elements. A flipped row inverts the matching column of
    ``pca.transform(X)``, which cascades into every downstream artifact.

    Canonical rule: ``components[i].sum() ≥ 0``. Unlike an ``argmax(|v|)``
    rule, picking the non-negative-sum representative is robust to
    single-element perturbations that cross two near-equal magnitudes.
    For rows whose sum is indistinguishable from zero (intrinsic ambiguity),
    fall back to ``argmax(|v|)`` for deterministic tie-breaking.

    Limitation: this only disambiguates per-row SIGN. It cannot align two
    differently-rotated bases of the same degenerate eigen-subspace — when
    ``explained_variance_[i] ≈ explained_variance_[i+1]``, sklearn versions
    may return arbitrary rotations within that subspace and the byte-equal
    gate will fall through to the 1e-6 element-wise PCA demotion path.
    """
    components = pca.components_
    row_sums = components.sum(axis=1)
    row_norms = np.linalg.norm(components, axis=1)
    near_zero = np.abs(row_sums) < 1e-10 * np.maximum(row_norms, 1e-20)
    signs = np.sign(row_sums).astype(components.dtype)
    if near_zero.any():
        max_abs_idx = np.argmax(np.abs(components), axis=1)
        row_indices = np.arange(components.shape[0])
        fallback_signs = np.sign(components[row_indices, max_abs_idx]).astype(components.dtype)
        signs = np.where(near_zero, fallback_signs, signs)
    signs[signs == 0] = 1.0
    pca.components_ = components * signs[:, None]
    # IncrementalPCA exposes no public API for rotating U alongside
    # components_, but the only consumers here (save_pca_params and
    # transform) read components_ directly, so in-place mutation suffices.
    return pca


def fit_incremental_pca(sample_vectors, n_components, batch_size=200000):
    """
    Fit PCA model using Incremental PCA for memory efficiency.

    Args:
        sample_vectors: Training vectors for fitting the PCA model
        n_components: Number of principal components to compute
        batch_size: Number of vectors processed per batch

    Returns:
        Fitted IncrementalPCA model (with canonical component signs).
    """
    n_samples = sample_vectors.shape[0]
    if n_samples < int(n_components):
        raise ValueError(f"PCA fitting requires at least n_components sample rows; got {n_samples} < {n_components}")
    ipca = IncrementalPCA(n_components=n_components, batch_size=min(batch_size, n_samples))
    for i in tqdm(range(0, n_samples, batch_size), desc="Training IncrementalPCA"):
        end_idx = min(i + batch_size, n_samples)
        ipca.partial_fit(sample_vectors[i:end_idx])
    _canonicalize_pca_sign(ipca)
    return ipca


def sample_vectors_from_fbin(filepath, sample_ratio=0.25, seed=None):
    """
    Sample a subset of vectors from an fbin file for PCA fitting.

    Args:
        filepath: Path to the fbin file
        sample_ratio: Fraction of vectors to sample
        seed: RNG seed. ``None`` preserves upstream Laser's non-deterministic
            behaviour (np.random.choice on the global state). Pass an int to
            make sample selection reproducible — required for any alignment
            or byte-equality gate across runs.

    Returns:
        Tuple of (all_vectors, sample_vectors)
    """
    vectors = read_fbin(filepath)
    n, d = vectors.shape
    sample_size = min(n, max(int(sample_ratio * n), d))

    if n > sample_size:
        rng = np.random.default_rng(seed) if seed is not None else np.random
        sample_indices = rng.choice(n, sample_size, replace=False)
        sample_indices = np.sort(sample_indices)

        sample_vecs = np.empty((sample_size, d), dtype=np.float32)
        with open(filepath, "rb") as f:
            for i, idx in enumerate(tqdm(sample_indices, desc="Reading sample vectors")):
                f.seek(8 + idx * d * 4)
                sample_vecs[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
        return vectors, sample_vecs
    return vectors, vectors


def pca_transform_and_save(vectors, pca, output_path, chunk_size=100000):
    """
    Transform vectors with PCA and save to fbin file in chunks.
    Supports resuming from an incomplete file.

    Args:
        vectors: Input vectors array
        pca: Fitted PCA model
        output_path: Output fbin file path
        chunk_size: Number of vectors per chunk
    """
    n, d = vectors.shape
    num_chunks = (n + chunk_size - 1) // chunk_size
    header_size = 8  # two int32s (n, d)
    bytes_per_vector = d * 4
    start_chunk = 0

    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > header_size:
            data_bytes = file_size - header_size
            completed_vectors = data_bytes // bytes_per_vector
            start_chunk = completed_vectors // chunk_size
            # Truncate to last complete chunk boundary
            valid_size = header_size + start_chunk * chunk_size * bytes_per_vector
            if file_size != valid_size:
                with open(output_path, "r+b") as f:
                    f.truncate(valid_size)
                print(f"Truncated partial chunk: {file_size} -> {valid_size} bytes")
            if start_chunk > 0:
                print(
                    f"Resuming PCA transform from chunk {start_chunk}/{num_chunks} "
                    f"({start_chunk * chunk_size:,} vectors done)"
                )

    if start_chunk == 0:
        with open(output_path, "wb") as f:
            np.array([n, d], dtype=np.int32).tofile(f)

    for chunk_idx in tqdm(
        range(start_chunk, num_chunks),
        desc="PCA transforming",
        initial=start_chunk,
        total=num_chunks,
    ):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        transformed = pca.transform(vectors[start_idx:end_idx])
        with open(output_path, "ab") as f:
            transformed.astype(np.float32).tofile(f)
