"""Logging, validation, and evaluation metrics."""

import logging
import sys
from typing import Optional

import numpy as np


# =============================================================================
# Logging
# =============================================================================

def setup_logging(level: str = "INFO", name: str = "mscoot") -> logging.Logger:
    """Set up logging with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    """Get or create the mscoot logger."""
    logger = logging.getLogger("mscoot")
    if not logger.handlers:
        return setup_logging()
    return logger


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_type_preservation(coupling: np.ndarray, types1: np.ndarray,
                              types2: np.ndarray) -> float:
    """Fraction of CPs matched to same-type CP (argmax of coupling row).

    Args:
        coupling: Coupling matrix (n1 x n2).
        types1: CP type array for source (n1,).
        types2: CP type array for target (n2,).

    Returns:
        Type preservation ratio in [0, 1].
    """
    n1 = coupling.shape[0]
    matches = np.argmax(coupling, axis=1)
    preserved = sum(1 for i in range(n1) if types1[i] == types2[matches[i]])
    return preserved / n1


def compute_spatial_coherence(coupling: np.ndarray, coords1: np.ndarray,
                              coords2: np.ndarray) -> float:
    """Mean Euclidean displacement of best-matched CPs.

    Args:
        coupling: Coupling matrix (n1 x n2).
        coords1: Source coordinates (n1, 2).
        coords2: Target coordinates (n2, 2).

    Returns:
        Mean displacement in coordinate units.
    """
    matches = np.argmax(coupling, axis=1)
    displacements = np.array([
        np.linalg.norm(coords1[i] - coords2[matches[i]])
        for i in range(len(matches))
    ])
    return displacements.mean()


def compute_coupling_entropy(coupling: np.ndarray) -> float:
    """Entropy of coupling matrix (lower = more concentrated).

    Args:
        coupling: Coupling matrix.

    Returns:
        Entropy value.
    """
    flat = coupling.flatten()
    flat = flat[flat > 1e-15]
    if len(flat) == 0:
        return 0.0
    return -np.sum(flat * np.log(flat))


def compute_type_preservation_mass(coupling: np.ndarray, types1: np.ndarray,
                                   types2: np.ndarray) -> float:
    """Fraction of coupling mass mapping between same-type nodes.

    Alternative metric that considers full coupling distribution, not just argmax.
    """
    same_type_mask = types1[:, np.newaxis] == types2[np.newaxis, :]
    same_type_mass = np.sum(coupling * same_type_mask)
    total_mass = np.sum(coupling)
    return same_type_mass / total_mass if total_mass > 0 else 0.0


# =============================================================================
# Validation
# =============================================================================

def validate_measure(measure: np.ndarray, name: str = "measure") -> None:
    """Validate that a probability measure sums to ~1."""
    if measure.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {measure.shape}")
    if np.any(measure < 0):
        raise ValueError(f"{name} contains negative values")
    total = measure.sum()
    if not np.isclose(total, 1.0, rtol=1e-5):
        raise ValueError(f"{name} does not sum to 1 (sum={total:.6f})")


def validate_coupling(matrix: np.ndarray, name: str = "coupling") -> None:
    """Validate basic properties of a coupling matrix."""
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {matrix.shape}")
    if np.any(matrix < 0):
        raise ValueError(f"{name} contains negative values")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
