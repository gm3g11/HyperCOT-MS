"""Distance matrix heatmap visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

from .style import apply_style, METHOD_COLORS


def plot_distance_heatmap(matrix: np.ndarray, title: str, output_path: Path,
                          cmap: str = 'viridis', label: str = 'Distance'):
    """Plot a single distance matrix as a heatmap.

    Args:
        matrix: Square distance matrix.
        title: Figure title.
        output_path: Path to save figure.
        cmap: Matplotlib colormap name.
        label: Colorbar label.
    """
    apply_style()
    n = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap=cmap, aspect='equal')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=10)

    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Timestep', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')

    tick_interval = max(1, n // 10) * 10
    ticks = list(range(0, n, tick_interval))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_multi_heatmaps(matrices: Dict[str, np.ndarray], output_path: Path,
                        suptitle: str = 'Distance Matrices'):
    """Plot multiple distance matrices side by side.

    Args:
        matrices: Dict of {method_name: distance_matrix}.
        output_path: Path to save figure.
        suptitle: Super title.
    """
    apply_style()
    n_methods = len(matrices)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    for ax, (name, mat) in zip(axes, matrices.items()):
        im = ax.imshow(mat, cmap='viridis', aspect='equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Timestep')

        n = mat.shape[0]
        tick_interval = max(1, n // 5) * 10
        ticks = list(range(0, n, tick_interval))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
