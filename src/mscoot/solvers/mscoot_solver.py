"""MS-COOT (Co-Optimal Transport on Morse-Smale hypergraphs) solver."""

import numpy as np
from ot import coot
from scipy.spatial.distance import cdist
from typing import Optional, Tuple


def compute_mscoot(
    omega1: np.ndarray, omega2: np.ndarray,
    mu1: np.ndarray, mu2: np.ndarray,
    nu1: np.ndarray, nu2: np.ndarray,
    types1: Optional[np.ndarray] = None,
    types2: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    epsilon: float = 0.01,
    coords1: Optional[np.ndarray] = None,
    coords2: Optional[np.ndarray] = None,
    spatial_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute HyperCOT (Co-Optimal Transport) between two MS complexes.

    Args:
        omega1: Hypernetwork function for source (n_cp1, n_regions1).
        omega2: Hypernetwork function for target (n_cp2, n_regions2).
        mu1: Node measure for source (n_cp1,).
        mu2: Node measure for target (n_cp2,).
        nu1: Hyperedge measure for source (n_regions1,).
        nu2: Hyperedge measure for target (n_regions2,).
        types1: CP types for source (optional, for type penalty).
        types2: CP types for target (optional, for type penalty).
        alpha: Type/spatial penalty weight (0 = structure only, 1 = type/spatial only).
        epsilon: Sinkhorn regularization (0 = exact EMD).
        coords1: Source CP coordinates (optional, for spatial penalty).
        coords2: Target CP coordinates (optional, for spatial penalty).
        spatial_weight: Weight for spatial cost (0 = type only).

    Returns:
        Tuple of (pi, xi, cost).
        pi: CP coupling matrix (n_cp1, n_cp2).
        xi: Region coupling matrix (n_regions1, n_regions2).
        cost: COOT distance value.
    """
    n1, m1 = omega1.shape
    n2, m2 = omega2.shape

    # Joint normalization (preserves relative distances between the two)
    scale = max(omega1.max(), omega2.max(), 1e-10)
    omega1_norm = omega1 / scale
    omega2_norm = omega2 / scale

    # Normalize marginals (clamp zeros to avoid Sinkhorn divide-by-zero)
    _eps = 1e-10
    mu1_n = np.maximum(mu1, _eps)
    mu1_n = mu1_n / mu1_n.sum()
    mu2_n = np.maximum(mu2, _eps)
    mu2_n = mu2_n / mu2_n.sum()
    nu1_n = np.maximum(nu1, _eps)
    nu1_n = nu1_n / nu1_n.sum()
    nu2_n = np.maximum(nu2, _eps)
    nu2_n = nu2_n / nu2_n.sum()

    # Build sample cost matrix M_samp
    M_samp = np.zeros((n1, n2))

    # Type penalty (vectorized)
    if types1 is not None and types2 is not None:
        M_samp = (types1[:, None] != types2[None, :]).astype(np.float64)

    # Spatial penalty
    if spatial_weight > 0 and coords1 is not None and coords2 is not None:
        M_spatial = cdist(coords1, coords2)
        M_spatial /= (M_spatial.max() + 1e-10)
        M_samp = M_samp + spatial_weight * M_spatial

    # Try log-domain Sinkhorn (stable for small epsilon), fall back to EMD
    pi, xi = None, None
    for eps in ([epsilon, 0] if epsilon > 0 else [0]):
        pi_try, xi_try, log = coot.co_optimal_transport(
            X=omega1_norm, Y=omega2_norm,
            wx_samp=mu1_n, wy_samp=mu2_n,
            wx_feat=nu1_n, wy_feat=nu2_n,
            epsilon=eps, alpha=alpha,
            M_samp=M_samp,
            method_sinkhorn='sinkhorn_log',
            nits_bcd=50, tol_bcd=1e-7,
            nits_ot=200, tol_sinkhorn=1e-7,
            log=True, verbose=False
        )
        if np.all(np.isfinite(pi_try)) and np.all(np.isfinite(xi_try)):
            pi, xi = pi_try, xi_try
            break

    if pi is None:
        raise RuntimeError("COOT failed with both Sinkhorn and EMD")

    # Compute COOT cost: quadratic structure term + linear sample penalty
    # cost = sum_{i,j,k,l} (omega1[i,k] - omega2[j,l])^2 * pi[i,j] * xi[k,l]
    #       + alpha * sum_{i,j} M_samp[i,j] * pi[i,j]
    #
    # Expanding (a-b)^2 = a^2 - 2ab + b^2 and factorizing over pi/xi:
    #   term_a = sum_{i,k} omega1[i,k]^2 * (sum_j pi[i,j]) * (sum_l xi[k,l])
    #   term_c = sum_{j,l} omega2[j,l]^2 * (sum_i pi[i,j]) * (sum_k xi[k,l])
    #   term_b = 2 * sum_{i,j,k,l} omega1[i,k] * omega2[j,l] * pi[i,j] * xi[k,l]
    #          = 2 * trace((omega1^T @ pi @ omega2) @ xi^T)
    #          = 2 * sum((pi^T @ omega1) * (omega2 @ xi^T))
    P1 = pi.sum(axis=1)    # (n1,) row sums of pi
    P2 = pi.sum(axis=0)    # (n2,) col sums of pi
    Q1 = xi.sum(axis=1)    # (m1,) row sums of xi
    Q2 = xi.sum(axis=0)    # (m2,) col sums of xi

    term_a = np.sum((omega1_norm ** 2) * (P1[:, None] * Q1[None, :]))
    term_c = np.sum((omega2_norm ** 2) * (P2[:, None] * Q2[None, :]))
    # cross term: sum((pi^T @ omega1) * (omega2 @ xi^T))
    term_b = 2.0 * np.sum((pi.T @ omega1_norm) * (omega2_norm @ xi.T))

    cost_structure = term_a + term_c - term_b
    cost_sample = alpha * np.sum(M_samp * pi)
    cost = cost_structure + cost_sample

    return pi, xi, cost
