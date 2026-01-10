"""
Utility functions for HyperCOT-MS project.

This module provides:
- Configuration loading and management
- Logging setup
- Path resolution
- Common data loading functions
- Error handling utilities
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


# =============================================================================
# Path Management
# =============================================================================

def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path: Absolute path to project root.
    """
    return Path(__file__).parent.resolve()


def resolve_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path relative to base directory or project root.

    Args:
        path: Path string (can be relative or absolute).
        base_dir: Base directory for relative paths. Defaults to project root.

    Returns:
        Path: Resolved absolute path.
    """
    if base_dir is None:
        base_dir = get_project_root()

    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (base_dir / path_obj).resolve()


# =============================================================================
# Configuration
# =============================================================================

_config_cache: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None, reload: bool = False) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config.yaml in project root.
        reload: Force reload even if cached.

    Returns:
        Dict containing configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid.
    """
    global _config_cache

    if _config_cache is not None and not reload:
        return _config_cache

    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    else:
        config_path = resolve_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    _config_cache = config
    return config


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value by dot-separated path.

    Args:
        key_path: Dot-separated path like 'hypercot.epsilon'.
        default: Default value if key not found.

    Returns:
        Configuration value or default.

    Example:
        >>> epsilon = get_config_value('hypercot.epsilon', 0.001)
    """
    config = load_config()
    keys = key_path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


# =============================================================================
# Logging
# =============================================================================

_logger: Optional[logging.Logger] = None


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    name: str = "hypercot"
) -> logging.Logger:
    """
    Set up logging with console and optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.
        name: Logger name.

    Returns:
        Configured logger.
    """
    global _logger

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = resolve_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """
    Get the configured logger, initializing if needed.

    Returns:
        Logger instance.
    """
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


# =============================================================================
# Data Loading
# =============================================================================

def load_critical_points(filepath: str) -> pd.DataFrame:
    """
    Load critical points CSV with validation.

    Args:
        filepath: Path to critical points CSV.

    Returns:
        DataFrame with critical point data.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required columns are missing.
    """
    path = resolve_path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Critical points file not found: {path}")

    df = pd.read_csv(path)

    required_cols = ['Points_0', 'Points_1', 'Points_2', 'CellDimension', 'data']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def load_separatrices(filepath: str) -> pd.DataFrame:
    """
    Load separatrices (edges) CSV with validation.

    Args:
        filepath: Path to separatrices CSV.

    Returns:
        DataFrame with edge data.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required columns are missing.
    """
    path = resolve_path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Separatrices file not found: {path}")

    df = pd.read_csv(path)

    required_cols = ['Points_0', 'Points_1', 'Points_2', 'CellId']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def load_hypergraph(filepath: str) -> pd.DataFrame:
    """
    Load hypergraph CSV with validation.

    Args:
        filepath: Path to hypergraph CSV.

    Returns:
        DataFrame with hyperedge data.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required columns are missing.
    """
    path = resolve_path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Hypergraph file not found: {path}")

    df = pd.read_csv(path)

    required_cols = ['min_id', 'max_id', 'boundary_saddles']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def load_coupling_matrix(filepath: str) -> np.ndarray:
    """
    Load coupling matrix from CSV.

    Args:
        filepath: Path to coupling matrix CSV.

    Returns:
        NumPy array with coupling values.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = resolve_path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Coupling matrix file not found: {path}")

    df = pd.read_csv(path, index_col=0)
    return df.values


# =============================================================================
# Error Handling
# =============================================================================

class HyperCOTError(Exception):
    """Base exception for HyperCOT errors."""
    pass


class DataValidationError(HyperCOTError):
    """Raised when data validation fails."""
    pass


class ConfigurationError(HyperCOTError):
    """Raised when configuration is invalid."""
    pass


def validate_coupling_matrix(matrix: np.ndarray, name: str = "coupling") -> None:
    """
    Validate that a coupling matrix has proper properties.

    Args:
        matrix: Coupling matrix to validate.
        name: Name for error messages.

    Raises:
        DataValidationError: If validation fails.
    """
    if not isinstance(matrix, np.ndarray):
        raise DataValidationError(f"{name} must be a numpy array")

    if matrix.ndim != 2:
        raise DataValidationError(f"{name} must be 2D, got shape {matrix.shape}")

    if np.any(matrix < 0):
        raise DataValidationError(f"{name} contains negative values")

    if np.any(np.isnan(matrix)):
        raise DataValidationError(f"{name} contains NaN values")

    if np.any(np.isinf(matrix)):
        raise DataValidationError(f"{name} contains infinite values")


def validate_measure(measure: np.ndarray, name: str = "measure") -> None:
    """
    Validate that a probability measure sums to 1.

    Args:
        measure: Probability measure to validate.
        name: Name for error messages.

    Raises:
        DataValidationError: If validation fails.
    """
    if not isinstance(measure, np.ndarray):
        raise DataValidationError(f"{name} must be a numpy array")

    if measure.ndim != 1:
        raise DataValidationError(f"{name} must be 1D, got shape {measure.shape}")

    if np.any(measure < 0):
        raise DataValidationError(f"{name} contains negative values")

    total = measure.sum()
    if not np.isclose(total, 1.0, rtol=1e-5):
        raise DataValidationError(f"{name} does not sum to 1 (sum={total:.6f})")


# =============================================================================
# Metrics
# =============================================================================

def compute_type_preservation(
    coupling: np.ndarray,
    types1: np.ndarray,
    types2: np.ndarray
) -> float:
    """
    Compute type preservation ratio for a coupling matrix.

    Type preservation is the fraction of coupling mass that maps
    between nodes of the same type (min-min, saddle-saddle, max-max).

    Args:
        coupling: Coupling matrix (n1 x n2).
        types1: Type array for first set (n1,).
        types2: Type array for second set (n2,).

    Returns:
        Type preservation ratio in [0, 1].
    """
    if coupling.shape[0] != len(types1) or coupling.shape[1] != len(types2):
        raise ValueError("Coupling matrix dimensions don't match type arrays")

    same_type_mask = types1[:, np.newaxis] == types2[np.newaxis, :]
    same_type_coupling = np.sum(coupling * same_type_mask)
    total_coupling = np.sum(coupling)

    if total_coupling == 0:
        return 0.0

    return same_type_coupling / total_coupling


def compute_entropy(coupling: np.ndarray) -> float:
    """
    Compute entropy of a coupling matrix (flattened as distribution).

    Args:
        coupling: Coupling matrix.

    Returns:
        Entropy value.
    """
    flat = coupling.flatten()
    flat = flat[flat > 0]  # Remove zeros
    if len(flat) == 0:
        return 0.0

    flat = flat / flat.sum()  # Normalize
    return -np.sum(flat * np.log(flat + 1e-12))
