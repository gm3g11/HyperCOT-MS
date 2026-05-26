"""Method comparison boxplot visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from .style import apply_style, METHOD_COLORS


def plot_method_comparison(metrics: Dict[str, Dict[str, np.ndarray]],
                           output_path: Path):
    """Plot boxplots comparing methods on type preservation, spatial coherence, entropy.

    Args:
        metrics: Dict of {method_name: {metric_name: values_array}}.
            Expected metric names: 'type_preservation', 'spatial_coherence', 'entropy'.
        output_path: Path to save figure.
    """
    apply_style()

    metric_configs = [
        ('type_preservation', 'Type Preservation (%)', 100),
        ('spatial_coherence', 'Spatial Coherence', 1),
        ('entropy', 'Coupling Entropy', 1),
    ]

    available = [(key, label, scale) for key, label, scale in metric_configs
                 if any(key in m for m in metrics.values())]

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    methods = list(metrics.keys())
    colors = [METHOD_COLORS.get(m, '#999999') for m in methods]

    for ax, (metric_key, ylabel, scale) in zip(axes, available):
        data = []
        labels = []
        for m in methods:
            if metric_key in metrics[m]:
                data.append(metrics[m][metric_key] * scale)
                labels.append(m)

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
