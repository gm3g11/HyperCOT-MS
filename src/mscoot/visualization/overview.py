"""Dataset overview and MS complex diagram visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from .style import apply_style, CP_COLORS, CP_LABELS


def plot_ms_complex(coords: np.ndarray, types: np.ndarray,
                    edges: list, output_path: Path,
                    scalar_field: Optional[np.ndarray] = None,
                    title: str = 'Morse-Smale Complex'):
    """Plot MS complex: CPs + separatrices on optional scalar field.

    Args:
        coords: CP coordinates (n, 2).
        types: CP types (n,).
        edges: Edge list [(i, j), ...].
        output_path: Path to save figure.
        scalar_field: Optional 2D scalar field for background.
        title: Figure title.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(14, 4))

    # Background scalar field
    if scalar_field is not None:
        ax.imshow(scalar_field.T, origin='lower', cmap='coolwarm', aspect='auto',
                  alpha=0.5)

    # Separatrices
    for i, j in edges:
        ax.plot([coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                'k-', alpha=0.3, linewidth=0.5)

    # CPs by type
    markers = {0: 'v', 1: 'D', 2: '^'}
    for cp_type in [0, 1, 2]:
        mask = types == cp_type
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=CP_COLORS[cp_type], marker=markers[cp_type],
                      s=40, zorder=5, edgecolors='k', linewidths=0.3,
                      label=CP_LABELS[cp_type])

    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
