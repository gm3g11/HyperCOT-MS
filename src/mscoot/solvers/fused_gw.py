"""Fused Gromov-Wasserstein (FGW) Distance solver."""

import numpy as np
import ot
from scipy.spatial.distance import cdist
from typing import Optional, Tuple


def compute_fgw(M: np.ndarray, C1: np.ndarray, C2: np.ndarray,
                alpha: float = 0.5,
                a: Optional[np.ndarray] = None,
                b: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Compute Fused Gromov-Wasserstein distance.

    Combines feature cost M with structural costs C1, C2.
    FGW = alpha * <M, G> + (1-alpha) * GW_loss (POT convention).

    Args:
        M: Feature cost matrix (n1, n2). E.g., spatial L2^2 / diagonal^2.
        C1: Structure cost for source (n1, n1). Normalized Dijkstra distances.
        C2: Structure cost for target (n2, n2). Normalized Dijkstra distances.
        alpha: Trade-off (0.5 = equal weight to feature and structure).
        a: Source weights (default: uniform).
        b: Target weights (default: uniform).

    Returns:
        Tuple of (coupling matrix, FGW distance value).
    """
    n1, n2 = M.shape

    if a is None:
        a = np.ones(n1) / n1
    if b is None:
        b = np.ones(n2) / n2

    coupling, log = ot.gromov.fused_gromov_wasserstein(
        M, C1, C2, a, b, loss_fun='square_loss', alpha=alpha, log=True
    )

    return coupling, log['fgw_dist']


def build_feature_cost(coords1: np.ndarray, coords2: np.ndarray,
                       diagonal: float) -> np.ndarray:
    """Build spatial feature cost matrix for FGW.

    Args:
        coords1: Source coordinates (n1, 2).
        coords2: Target coordinates (n2, 2).
        diagonal: Domain diagonal for normalization.

    Returns:
        Cost matrix (n1, n2): L2 / diagonal.
    """
    return cdist(coords1, coords2, metric='euclidean') / diagonal
