"""Evaluation utilities for nearest neighbor search quality."""

import numpy as np


def compute_distances(base, query, ids):
    """Compute L2 distances between a query and base vectors at given indices."""
    return np.linalg.norm(base[ids] - query, axis=1)


def calc_distance_ratio(distances, gt_distances):
    """
    Calculate the distance ratio between search results and ground truth.

    A ratio of 1.0 means perfect results; higher means worse quality.
    """
    distances = sorted(distances)
    gt_distances = sorted(gt_distances)
    ratios = [d1 / d2 for d1, d2 in zip(distances, gt_distances) if d2 > 1e-4]
    if len(ratios) == 0:
        return 1 * len(gt_distances)
    return sum(ratios) / len(ratios) * len(gt_distances)


def compute_recall(results, gt, k):
    """
    Compute recall@k for a set of queries.

    Args:
        results: List of result ID arrays, one per query
        gt: Ground truth array (nq, k_gt)
        k: Number of nearest neighbors to evaluate

    Returns:
        Recall as a fraction in [0, 1]
    """
    total_correct = 0
    nq = len(results)
    for i in range(nq):
        res_set = set(results[i])
        for j in range(k):
            if gt[i][j] in res_set:
                total_correct += 1
    return total_correct / (nq * k)
