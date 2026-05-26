"""Hypernetwork function (omega) computation via augmented graph + Dijkstra."""

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from typing import List, Tuple

from ..graph.virtual_center import compute_virtual_center


def find_adjacent_regions(hyperedge_members: List[List[int]]) -> List[Tuple[int, int]]:
    """Find pairs of adjacent regions (sharing 2+ boundary CPs)."""
    n = len(hyperedge_members)
    adjacency = []
    for i in range(n):
        set_i = set(hyperedge_members[i])
        for j in range(i + 1, n):
            if len(set_i & set(hyperedge_members[j])) >= 2:
                adjacency.append((i, j))
    return adjacency


def compute_omega(coords: np.ndarray, types: np.ndarray,
                  edges: List[Tuple[int, int]],
                  hyperedge_members: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute omega matrix: shortest path from each CP to each virtual center.

    Builds augmented graph with 3 edge types:
    1. CP<->CP (separatrices, L2 weight)
    2. CP<->VC (boundary membership, L2 weight)
    3. VC<->VC (adjacent regions sharing 2+ boundary CPs, L2 weight)

    Args:
        coords: CP coordinates (n_cp, 3) — include z even if 2D (set z=0).
        types: CP types (n_cp,).
        edges: Separatrix edge list [(i,j), ...].
        hyperedge_members: List of member lists per hyperedge.

    Returns:
        Tuple of (omega, virtual_centers).
        omega: (n_cp, n_regions) normalized to [0, 1].
        virtual_centers: (n_regions, 3).
    """
    n_cp = len(coords)
    n_hyper = len(hyperedge_members)

    if n_hyper == 0:
        return np.zeros((n_cp, 0)), np.zeros((0, 3))

    # Compute virtual centers
    virtual_centers = np.array([
        compute_virtual_center(coords, types, members)
        for members in hyperedge_members
    ])

    n_total = n_cp + n_hyper
    edge_list = []

    # 1. CP<->CP edges (separatrices)
    for i, j in edges:
        dist = np.linalg.norm(coords[i] - coords[j])
        edge_list.append((i, j, dist))
        edge_list.append((j, i, dist))

    # 2. CP<->VC edges (boundary membership)
    for h_idx, members in enumerate(hyperedge_members):
        vc_node = n_cp + h_idx
        vc_pos = virtual_centers[h_idx]
        for cp_idx in members:
            if cp_idx < n_cp:
                dist = np.linalg.norm(coords[cp_idx] - vc_pos)
                edge_list.append((cp_idx, vc_node, dist))
                edge_list.append((vc_node, cp_idx, dist))

    # 3. VC<->VC edges (adjacent regions)
    for i, j in find_adjacent_regions(hyperedge_members):
        vc_i = n_cp + i
        vc_j = n_cp + j
        dist = np.linalg.norm(virtual_centers[i] - virtual_centers[j])
        edge_list.append((vc_i, vc_j, dist))
        edge_list.append((vc_j, vc_i, dist))

    # Build sparse matrix and run Dijkstra
    if edge_list:
        rows, cols, weights = zip(*edge_list)
        adj = csr_matrix((weights, (rows, cols)), shape=(n_total, n_total))
    else:
        adj = csr_matrix((n_total, n_total))

    dist_matrix = dijkstra(adj, directed=False, indices=range(n_cp))

    # Extract CP->VC distances
    omega = dist_matrix[:, n_cp:n_cp + n_hyper]

    # Replace inf with large finite value
    finite_mask = np.isfinite(omega) & (omega > 0)
    max_finite = omega[finite_mask].max() if np.any(finite_mask) else 1.0
    omega[~np.isfinite(omega)] = max_finite * 2

    # Normalize to [0, 1]
    omega = omega / (omega.max() + 1e-8)

    return omega, virtual_centers
