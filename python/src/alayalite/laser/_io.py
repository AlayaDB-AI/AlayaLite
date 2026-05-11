# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Binary vector file I/O (fbin / ibin format)."""

import numpy as np


def read_fbin(filename, start_idx=0, chunk_size=None, use_mmap=True):
    """Read *.fbin file (float32 vectors).

    Args:
        filename: path to *.fbin file.
        start_idx: start reading vectors from this index.
        chunk_size: number of vectors to read; None means all.
        use_mmap: use memory-mapped file for efficient random access.
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
    nvecs, dim = int(nvecs), int(dim)

    if chunk_size is None:
        chunk_size = nvecs - start_idx
    else:
        chunk_size = min(chunk_size, nvecs - start_idx)

    if use_mmap:
        return np.memmap(
            filename,
            dtype=np.float32,
            mode="r",
            offset=8 + start_idx * dim * 4,
            shape=(chunk_size, dim),
        )
    arr = np.fromfile(
        filename,
        dtype=np.float32,
        offset=8 + start_idx * dim * 4,
        count=chunk_size * dim,
    )
    return arr.reshape(chunk_size, dim)


def read_ibin(filename, start_idx=0, chunk_size=None):
    """Read *.ibin file (int32 vectors)."""
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def write_fbin(filename, arr):
    """Write *.fbin file (float32 vectors)."""
    nvecs, dim = arr.shape
    with open(filename, "wb") as f:
        np.array([nvecs, dim], dtype=np.int32).tofile(f)
        np.ascontiguousarray(arr, dtype=np.float32).tofile(f)


def write_ibin(filename, arr):
    """Write *.ibin file (int32 vectors)."""
    nvecs, dim = arr.shape
    with open(filename, "wb") as f:
        np.array([nvecs, dim], dtype=np.int32).tofile(f)
        np.ascontiguousarray(arr, dtype=np.int32).tofile(f)
