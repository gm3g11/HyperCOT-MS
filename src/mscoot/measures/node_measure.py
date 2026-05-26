"""Node measure (mu) computation from persistence images."""

import numpy as np
from collections import defaultdict
from scipy.stats import multivariate_normal
from typing import List, Tuple

from .persistence import compute_extended_persistence, compute_persistence_image


def compute_mu(scalars: np.ndarray, edges: List[Tuple[int, int]],
               sigma: float = 0.3, resolution: int = 100) -> np.ndarray:
    """Compute node measure mu for all CPs.

    Each CP's weight is its contribution to the persistence image,
    reflecting its topological importance.

    Scalars are normalized to [0, 1] internally so that the fixed sigma
    parameter produces meaningful PI resolution regardless of the original
    scalar range.

    Args:
        scalars: Scalar values at each CP.
        edges: List of (i, j) edge pairs.
        sigma: Persistence image kernel width (relative to [0,1] range).
        resolution: PI grid resolution.

    Returns:
        mu array (n_cp,), normalized to sum to 1.
    """
    n_cp = len(scalars)

    # Normalize scalars to [0, 1] so sigma=0.3 is always meaningful
    s_min, s_max = scalars.min(), scalars.max()
    s_range = s_max - s_min
    if s_range > 1e-10:
        scalars_norm = (scalars - s_min) / s_range
    else:
        return np.ones(n_cp) / n_cp

    pairs_bd = compute_extended_persistence(scalars_norm, edges)
    I_f, finite_pairs_bp, grid_centers = compute_persistence_image(
        pairs_bd, resolution, resolution, sigma
    )

    if grid_centers is None or len(pairs_bd) == 0:
        return np.ones(n_cp) / n_cp

    # Map vertices to (birth, persistence) points using normalized values
    tolerance = 1e-5
    vertex_to_bp = {}
    value_to_vertices = defaultdict(list)
    for i, fv in enumerate(scalars_norm):
        value_to_vertices[fv].append(i)

    possible_vals = np.array(list(value_to_vertices.keys()))

    for b, d in pairs_bd:
        p = abs(d - b)
        point_bp = (b, p)

        birth_matches = np.where(np.isclose(possible_vals, b, atol=tolerance))[0]
        birth_verts = [v for idx in birth_matches for v in value_to_vertices[possible_vals[idx]]]

        death_matches = np.where(np.isclose(possible_vals, d, atol=tolerance))[0]
        death_verts = [v for idx in death_matches for v in value_to_vertices[possible_vals[idx]]]

        for v in birth_verts + death_verts:
            if v not in vertex_to_bp:
                vertex_to_bp[v] = point_bp

    # Compute contributions
    cov = (sigma ** 2) * np.eye(2)
    contributions = np.zeros(n_cp)
    for v in range(n_cp):
        if v in vertex_to_bp:
            b_v, p_v = vertex_to_bp[v]
            gauss_v = multivariate_normal(mean=[b_v, p_v], cov=cov)
            contributions[v] = np.sum(I_f * gauss_v.pdf(grid_centers))

    total = contributions.sum()
    if total > 1e-9:
        mu = contributions / total
    else:
        return np.ones(n_cp) / n_cp

    # Unpaired CPs (not in any persistence pair) get a small epsilon weight
    # so they are not invisible to transport. This affects ~2 saddles per mesh
    # in degree-1 max regions (3-CP hyperedges).
    zero_mask = mu < 1e-9
    if zero_mask.any():
        mu[zero_mask] = 1e-3
        mu = mu / mu.sum()

    return mu


def apply_mu_floor(mu: np.ndarray, floor_fraction: float = 0.20) -> np.ndarray:
    """Apply minimum weight floor: mu_new = (1-f)*mu + f*uniform.

    Prevents zero-weight CPs from being ignored in transport.

    Args:
        mu: Original mu array.
        floor_fraction: Fraction of uniform distribution to mix in.

    Returns:
        Floored mu, normalized to sum to 1.
    """
    if floor_fraction <= 0:
        return mu.copy()
    n = len(mu)
    mu_new = (1 - floor_fraction) * mu + floor_fraction * np.ones(n) / n
    return mu_new / mu_new.sum()
