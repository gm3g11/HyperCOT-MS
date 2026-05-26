"""CP correspondence overlay visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Optional

from .style import apply_style, CP_COLORS


def plot_cp_correspondence(coords1: np.ndarray, coords2: np.ndarray,
                           types1: np.ndarray, types2: np.ndarray,
                           coupling: np.ndarray,
                           output_path: Path,
                           title: str = 'CP Correspondence',
                           top_k: int = 20,
                           offset_y: float = 0):
    """Plot CP matching with lines connecting matched pairs.

    Args:
        coords1: Source CP coordinates (n1, 2).
        coords2: Target CP coordinates (n2, 2).
        types1: Source CP types.
        types2: Target CP types.
        coupling: Coupling matrix (n1, n2).
        output_path: Path to save figure.
        title: Figure title.
        top_k: Number of top matches to show lines for.
        offset_y: Vertical offset for target CPs.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Offset target CPs vertically for visibility
    coords2_shifted = coords2.copy()
    if offset_y != 0:
        coords2_shifted[:, 1] += offset_y

    # Draw matching lines (top-k by coupling strength)
    matches = np.argmax(coupling, axis=1)
    match_strengths = np.array([coupling[i, matches[i]] for i in range(len(matches))])
    top_indices = np.argsort(match_strengths)[-top_k:]

    norm = Normalize(vmin=match_strengths[top_indices].min(),
                     vmax=match_strengths[top_indices].max())

    for idx in top_indices:
        j = matches[idx]
        strength = match_strengths[idx]
        alpha = 0.3 + 0.7 * norm(strength)
        ax.plot([coords1[idx, 0], coords2_shifted[j, 0]],
                [coords1[idx, 1], coords2_shifted[j, 1]],
                color='gray', alpha=alpha, linewidth=0.8)

    # Plot CPs
    for cp_type, color in CP_COLORS.items():
        mask1 = types1 == cp_type
        mask2 = types2 == cp_type
        if mask1.any():
            ax.scatter(coords1[mask1, 0], coords1[mask1, 1],
                      c=color, s=30, zorder=5, edgecolors='k', linewidths=0.3)
        if mask2.any():
            ax.scatter(coords2_shifted[mask2, 0], coords2_shifted[mask2, 1],
                      c=color, s=30, zorder=5, edgecolors='k', linewidths=0.3,
                      marker='s')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_coupling_matrix(coupling: np.ndarray, output_path: Path,
                         title: str = 'Coupling Matrix',
                         xlabel: str = 'Target CP',
                         ylabel: str = 'Source CP'):
    """Plot a coupling matrix as a heatmap."""
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(coupling, cmap='Reds', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
