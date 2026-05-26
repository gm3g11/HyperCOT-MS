"""Region tracking visualization across timesteps."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Optional

from .style import apply_style


def plot_region_tracking(region_data: List[Dict], xi_matrices: List[np.ndarray],
                         output_path: Path, title: str = 'Region Tracking'):
    """Plot region tracking across multiple timesteps with consistent colors.

    Regions matched via xi coupling get the same color across frames.

    Args:
        region_data: List of dicts per timestep, each with:
            'coords': region boundary coordinates or centroid positions
            'n_regions': number of regions
        xi_matrices: List of xi coupling matrices between consecutive frames.
        output_path: Path to save figure.
        title: Figure title.
    """
    apply_style()

    n_frames = len(region_data)
    fig, axes = plt.subplots(1, n_frames, figsize=(5 * n_frames, 4))
    if n_frames == 1:
        axes = [axes]

    # Generate distinct colors
    base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    extra_colors = plt.cm.Set3(np.linspace(0, 1, 12))
    all_colors = np.vstack([base_colors, extra_colors])

    # Track color assignments across frames
    n_regions_0 = region_data[0]['n_regions']
    color_map = {0: {i: all_colors[i % len(all_colors)] for i in range(n_regions_0)}}

    for frame_idx in range(1, n_frames):
        xi = xi_matrices[frame_idx - 1]
        prev_map = color_map[frame_idx - 1]
        curr_map = {}

        # Match regions via xi argmax
        for j in range(xi.shape[1]):
            best_i = np.argmax(xi[:, j])
            if best_i in prev_map:
                curr_map[j] = prev_map[best_i]
            else:
                curr_map[j] = all_colors[j % len(all_colors)]

        color_map[frame_idx] = curr_map

    # Plot each frame
    for frame_idx, (ax, data) in enumerate(zip(axes, region_data)):
        colors = color_map[frame_idx]
        n_reg = data['n_regions']

        for r in range(n_reg):
            color = colors.get(r, [0.8, 0.8, 0.8, 1.0])
            # Simple visualization: colored bar per region
            ax.barh(r, 1, color=color, edgecolor='k', linewidth=0.3)

        ax.set_ylim(-0.5, n_reg - 0.5)
        ax.set_xlim(0, 1)
        ax.set_yticks(range(n_reg))
        ax.set_xlabel(f'Frame {frame_idx}')
        ax.set_ylabel('Region')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_xi_matrices(xi_list: List[np.ndarray], labels: List[str],
                     output_path: Path, title: str = 'Region Coupling Matrices'):
    """Plot xi (region coupling) matrices for multiple consecutive pairs.

    Args:
        xi_list: List of xi matrices.
        labels: List of pair labels (e.g., ['0->1', '1->2']).
        output_path: Path to save figure.
        title: Figure title.
    """
    apply_style()

    n = len(xi_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, xi, label in zip(axes, xi_list, labels):
        im = ax.imshow(xi, cmap='Oranges', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Target Region')
        ax.set_ylabel('Source Region')

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
