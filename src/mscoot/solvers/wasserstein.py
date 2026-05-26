"""Wasserstein Distance (WD) solver."""

import numpy as np
import ot
from scipy.spatial.distance import cdist
from typing import Optional, Tuple


def compute_wd(coords1: np.ndarray, coords2: np.ndarray, diagonal: float,
               a: Optional[np.ndarray] = None,
               b: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Compute Wasserstein distance between two CP sets.

    Uses spatial L2 cost normalized by domain diagonal.

    Args:
        coords1: Source coordinates (n1, 2).
        coords2: Target coordinates (n2, 2).
        diagonal: Domain diagonal for normalization.
        a: Source weights (default: uniform).
        b: Target weights (default: uniform).

    Returns:
        Tuple of (coupling matrix, distance value).
    """
    n1, n2 = len(coords1), len(coords2)

    if a is None:
        a = np.ones(n1) / n1
    if b is None:
        b = np.ones(n2) / n2

    # Cost matrix: L2 / diagonal
    C = cdist(coords1, coords2, metric='euclidean') / diagonal

    coupling = ot.emd(a, b, C)
    distance = np.sum(coupling * C)

    return coupling, distance
