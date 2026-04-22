import numpy as np


def ivecs_read(filename):
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"\t{filename} readed")
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(filename):
    return ivecs_read(filename).view("float32")


def ivecs_write(filename, m):
    print(f"Writing File - {filename}")
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype="int32")
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(filename)
    print(f"\t{filename} wrote")


def fvecs_write(filename, m):
    m = m.astype("float32")
    ivecs_write(filename, m.view("int32"))


###
# https://github.com/harsha-simhadri/big-ann-benchmarks
# MIT License

# Copyright (c) 2021 Martin Aumüller

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###

# def read_fbin(filename, start_idx=0, chunk_size=None):
#     """ Read *.fbin file that contains float32 vectors
#     Args:
#         :param filename (str): path to *.fbin file
#         :param start_idx (int): start reading vectors from this index
#         :param chunk_size (int): number of vectors to read.
#                                  If None, read all vectors
#     Returns:
#         Array of float32 vectors (numpy.ndarray)
#     """
#     with open(filename, "rb") as f:
#         nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
#         nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
#         arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
#                           offset=start_idx * 4 * dim)
#     return arr.reshape(nvecs, dim)


def read_fbin(filename, start_idx=0, chunk_size=None, use_mmap=True):
    """
    Read *.fbin file that contains float32 vectors.
    Args:
        :param filename (str): path to *.fbin file.
        :param start_idx (int): start reading vectors from this index.
        :param chunk_size (int): number of vectors to read. If None, read all vectors.
        :param use_mmap (bool): Whether to use memory-mapped file for faster access.
    Returns:
        Array of float32 vectors (numpy.ndarray).
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
    nvecs, dim = int(nvecs), int(dim)

    if chunk_size is None:
        chunk_size = nvecs - start_idx
    else:
        chunk_size = min(chunk_size, nvecs - start_idx)

    if use_mmap:
        # Use memory-mapped file for efficient random access
        mmap = np.memmap(
            filename,
            dtype=np.float32,
            mode="r",
            offset=8 + start_idx * dim * 4,
            shape=(chunk_size, dim),
        )
        return mmap
    else:
        # Read data directly into memory
        arr = np.fromfile(
            filename,
            dtype=np.float32,
            offset=8 + start_idx * dim * 4,
            count=chunk_size * dim,
        )
        return arr.reshape(chunk_size, dim)


def read_ibin(filename, start_idx=0, chunk_size=None):
    """Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(
            f, count=nvecs * dim, dtype=np.int32, offset=start_idx * 4 * dim
        )
    return arr.reshape(nvecs, dim)


def write_fbin(filename, arr):
    """Write *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param arr (numpy.ndarray): array of float32 vectors to be written
    """
    nvecs, dim = arr.shape
    with open(filename, "wb") as f:
        # Write number of vectors and dimension
        np.array([nvecs, dim], dtype=np.int32).tofile(f)
        # Write the vectors data
        arr.astype(np.float32).tofile(f)


def write_ibin(filename, arr):
    """Write *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param arr (numpy.ndarray): array of int32 vectors to be written
    """
    nvecs, dim = arr.shape
    with open(filename, "wb") as f:
        # Write number of vectors and dimension
        np.array([nvecs, dim], dtype=np.int32).tofile(f)
        # Write the vectors data
        arr.astype(np.int32).tofile(f)
