"""Ground truth computation for nearest neighbor evaluation."""

import numpy as np

from alayalite.laser.io import fvecs_read, ivecs_write
from alayalite.laser.preprocess import normalize


def compute_groundtruth(base, query, k=1000, metric="l2"):
    """
    Compute exact k-nearest neighbors by brute-force.

    Args:
        base: Base vectors array (n, d)
        query: Query vectors array (nq, d)
        k: Number of nearest neighbors to find
        metric: Distance metric ("l2" or "angular")

    Returns:
        Ground truth indices array (nq, k)
    """
    if metric == "angular":
        base = normalize(base)
        query = normalize(query)

    gt = []
    for q in query:
        distances = np.linalg.norm(base - q, axis=1)
        gt.append(np.argsort(distances)[:k])
    return np.array(gt)


def compute_and_save_groundtruth(base_path, query_path, output_path, k=1000, metric="l2"):
    """
    Compute ground truth from fvecs files and save as ivecs.

    Args:
        base_path: Path to base vectors (fvecs format)
        query_path: Path to query vectors (fvecs format)
        output_path: Output path for ground truth (ivecs format)
        k: Number of nearest neighbors
        metric: Distance metric ("l2" or "angular")
    """
    base = fvecs_read(base_path)
    query = fvecs_read(query_path)
    gt = compute_groundtruth(base, query, k=k, metric=metric)
    ivecs_write(output_path, gt)
