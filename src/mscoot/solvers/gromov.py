"""Gromov-Wasserstein Distance (GWD) solver."""

import numpy as np
import ot
from typing import Optional, Tuple


def compute_gwd(D1: np.ndarray, D2: np.ndarray,
                a: Optional[np.ndarray] = None,
                b: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Compute Gromov-Wasserstein distance between two metric spaces.

    Args:
        D1: Pairwise distance matrix for source (n1, n1). Should be normalized to [0,1].
        D2: Pairwise distance matrix for target (n2, n2). Should be normalized to [0,1].
        a: Source weights (default: uniform).
        b: Target weights (default: uniform).

    Returns:
        Tuple of (coupling matrix, GWD value).
    """
    n1, n2 = D1.shape[0], D2.shape[0]

    if a is None:
        a = np.ones(n1) / n1
    if b is None:
        b = np.ones(n2) / n2

    coupling, log = ot.gromov.gromov_wasserstein(
        D1, D2, a, b, loss_fun='square_loss', log=True
    )

    return coupling, log['gw_dist']
