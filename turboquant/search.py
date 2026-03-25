"""Search utilities for raw and compressed vector databases."""

from __future__ import annotations

import numpy as np


def brute_force_search(
    queries: np.ndarray,
    database: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth top-k inner product search on raw (uncompressed) vectors.

    Args:
        queries: (nq, d) float32 query vectors.
        database: (n, d) float32 database vectors.
        k: Number of top results.

    Returns:
        (indices, scores) each of shape (nq, k), descending order.
    """
    queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
    database = np.asarray(database, dtype=np.float32)

    scores = queries @ database.T  # (nq, n)

    if k >= scores.shape[1]:
        indices = np.argsort(-scores, axis=1)
        sorted_scores = np.take_along_axis(scores, indices, axis=1)
        return indices, sorted_scores

    top_k_unsorted = np.argpartition(-scores, k, axis=1)[:, :k]
    top_k_scores = np.take_along_axis(scores, top_k_unsorted, axis=1)
    sort_order = np.argsort(-top_k_scores, axis=1)
    top_k_sorted = np.take_along_axis(top_k_unsorted, sort_order, axis=1)
    top_k_scores_sorted = np.take_along_axis(top_k_scores, sort_order, axis=1)

    return top_k_sorted, top_k_scores_sorted


def recall_at_k(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    k: int | None = None,
) -> float:
    """Compute Recall@k — fraction of true top-k found in predicted top-k.

    Args:
        ground_truth: (nq, k_gt) array of true top-k indices.
        predicted: (nq, k_pred) array of predicted top-k indices.
        k: If specified, use only the first k columns of each.

    Returns:
        Average recall across all queries (float in [0, 1]).
    """
    if k is not None:
        ground_truth = ground_truth[:, :k]
        predicted = predicted[:, :k]

    nq = ground_truth.shape[0]
    recall_sum = 0.0
    for i in range(nq):
        gt_set = set(ground_truth[i])
        pred_set = set(predicted[i])
        recall_sum += len(gt_set & pred_set) / len(gt_set)

    return recall_sum / nq
