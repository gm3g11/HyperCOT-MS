"""CP adjacency matrix construction from separatrices."""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


def build_adjacency_from_edges(n_cp: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Build binary adjacency matrix from edge list.

    Args:
        n_cp: Number of critical points.
        edges: List of (i, j) undirected edge pairs (local indices).

    Returns:
        Binary adjacency matrix (n_cp x n_cp).
    """
    adjacency = np.zeros((n_cp, n_cp))
    for i, j in edges:
        if i < n_cp and j < n_cp and i != j:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
    return adjacency


def build_weighted_adjacency(adjacency: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Build Euclidean-weighted adjacency (0 -> inf for non-edges).

    Args:
        adjacency: Binary adjacency matrix (n x n).
        coords: Coordinate array (n, d).

    Returns:
        Weighted adjacency with inf for non-edges, 0 on diagonal.
    """
    from scipy.spatial.distance import cdist
    euclidean = cdist(coords, coords, metric='euclidean')
    weighted = np.where(adjacency > 0, euclidean, np.inf)
    np.fill_diagonal(weighted, 0)
    return weighted


def compute_graph_distances(adjacency: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Compute normalized Dijkstra shortest path distance matrix.

    Used by GWD and FGW solvers. Edge weights are Euclidean distances.
    Self-normalized to [0, 1] for pure structural comparison.

    Args:
        adjacency: Binary adjacency matrix (n x n).
        coords: Coordinate array (n, 2).

    Returns:
        Normalized distance matrix in [0, 1].
    """
    from scipy.sparse.csgraph import shortest_path

    weighted = build_weighted_adjacency(adjacency, coords)
    D = shortest_path(weighted, directed=False, method='D')

    # Handle disconnected components
    max_dist = D[np.isfinite(D)].max() if np.any(np.isfinite(D)) else 1
    D[np.isinf(D)] = max_dist * 2

    # Self-normalize to [0, 1] for pure structural comparison
    D = D / (D.max() + 1e-8)
    return D
