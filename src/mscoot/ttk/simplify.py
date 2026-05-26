"""Persistence-based simplification utilities."""

import numpy as np
from typing import List, Tuple


def filter_by_persistence(pairs: List[Tuple[float, float]],
                          threshold: float) -> List[Tuple[float, float]]:
    """Filter persistence pairs by absolute persistence threshold.

    Args:
        pairs: List of (birth, death) pairs.
        threshold: Minimum persistence to keep.

    Returns:
        Filtered list of pairs.
    """
    return [(b, d) for b, d in pairs if abs(d - b) >= threshold]


def compute_persistence_threshold(pairs: List[Tuple[float, float]],
                                  fraction: float = 0.01) -> float:
    """Compute absolute persistence threshold as fraction of max persistence.

    Args:
        pairs: List of (birth, death) pairs.
        fraction: Fraction of maximum persistence.

    Returns:
        Absolute threshold value.
    """
    if not pairs:
        return 0.0
    max_pers = max(abs(d - b) for b, d in pairs)
    return fraction * max_pers
