"""Extended persistence computation and persistence images."""

import numpy as np
import gudhi as gd
from scipy.stats import multivariate_normal
from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='gudhi')


def compute_extended_persistence(
    scalars: np.ndarray, edges: List[Tuple[int, int]]
) -> List[Tuple[float, float]]:
    """Compute extended persistence diagram from scalar filtration.

    Args:
        scalars: Scalar values at each vertex.
        edges: List of (i, j) edge pairs.

    Returns:
        List of (birth, death) pairs from all persistence types.
    """
    n = len(scalars)
    st = gd.SimplexTree()

    for i in range(n):
        st.insert([i], filtration=scalars[i])

    for i, j in edges:
        f_edge = max(scalars[i], scalars[j])
        st.insert([i, j], filtration=f_edge)

    st.make_filtration_non_decreasing()
    st.extend_filtration()
    dgms = st.extended_persistence()

    tolerance = 1e-5
    pairs = []
    for dgm in dgms:
        for dim, (b, d) in dgm:
            if np.isfinite(b) and np.isfinite(d) and abs(b - d) > tolerance:
                pairs.append((b, d))

    return pairs


def compute_persistence_image(
    pairs_bd: List[Tuple[float, float]],
    nx_res: int = 100, ny_res: int = 100, sigma: float = 0.3
) -> Tuple[np.ndarray, List[Tuple[float, float]], Optional[np.ndarray]]:
    """Compute persistence image from birth-death pairs.

    Args:
        pairs_bd: List of (birth, death) pairs.
        nx_res: Grid resolution in birth direction.
        ny_res: Grid resolution in persistence direction.
        sigma: Gaussian kernel bandwidth.

    Returns:
        Tuple of (image_flat, finite_pairs_bp, grid_centers).
        image_flat is shape (nx_res * ny_res,).
        grid_centers is shape (nx_res * ny_res, 2) or None if no valid pairs.
    """
    if not pairs_bd:
        return np.zeros(nx_res * ny_res), [], None

    tolerance = 1e-5
    finite_pairs_bp = [(b, abs(d - b)) for b, d in pairs_bd if abs(d - b) > tolerance]

    if not finite_pairs_bp:
        return np.zeros(nx_res * ny_res), [], None

    weights = [p for _, p in finite_pairs_bp]
    bs = [b for b, _ in finite_pairs_bp]
    ps = [p for _, p in finite_pairs_bp]

    min_b, max_b = min(bs), max(bs)
    min_p, max_p = min(ps), max(ps)

    buf_b = 3 * sigma + 0.05 * (max_b - min_b) if max_b > min_b else 3 * sigma
    buf_p = 3 * sigma + 0.05 * (max_p - min_p) if max_p > min_p else 3 * sigma

    birth_range = np.linspace(min_b - buf_b, max_b + buf_b, nx_res)
    pers_range = np.linspace(max(0, min_p - buf_p), max_p + buf_p, ny_res)

    X, Y = np.meshgrid(birth_range, pers_range)
    grid_centers = np.vstack([X.ravel(), Y.ravel()]).T

    I_f = np.zeros(nx_res * ny_res)
    cov = (sigma ** 2) * np.eye(2)
    for (b_val, p_val), w in zip(finite_pairs_bp, weights):
        gauss = multivariate_normal(mean=(b_val, p_val), cov=cov)
        I_f += w * gauss.pdf(grid_centers)

    return I_f, finite_pairs_bp, grid_centers
