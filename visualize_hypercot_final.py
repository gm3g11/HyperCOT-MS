"""
HyperCOT Visualization with Color = Correspondence Principle

Visualization approach:
- Clean regions: Unique color + ID label (identity)
- Noisy regions: Inherit color from best match + show matched clean ID
- Spurious regions: Gray (no strong match)
- Opacity: Proportional to coupling confidence

Two main figures:
1. hypercot_detailed_correspondence.png - Hyperedge (region) correspondence
2. hypercot_node_correspondence.png - Node (CP) correspondence with displacement arrows
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
# linear_sum_assignment no longer needed - showing full transport plan
import os
from pathlib import Path

BASE_PATH = Path(__file__).parent.resolve()

# Thresholds and parameters
SPURIOUS_THRESHOLD = 0.002    # Absolute threshold for spurious detection
OPACITY_MIN = 0.4             # Minimum opacity for weak matches
OPACITY_MAX = 0.9             # Maximum opacity for strong matches

# CP visualization
CP_COLORS = {0: '#2166ac', 1: '#4daf4a', 2: '#e41a1c'}
CP_MARKERS = {0: 'o', 1: 's', 2: '^'}


def load_data():
    """Load all required data."""
    xi = pd.read_csv(os.path.join(BASE_PATH, "hypercot_xi.csv"), index_col=0).values
    pi = pd.read_csv(os.path.join(BASE_PATH, "hypercot_pi.csv"), index_col=0).values
    clean_cp = pd.read_csv(os.path.join(BASE_PATH, "clean_critical_points.csv"))
    noisy_cp = pd.read_csv(os.path.join(BASE_PATH, "noisy_critical_points.csv"))
    clean_vc = pd.read_csv(os.path.join(BASE_PATH, "clean_virtual_centers.csv"))
    noisy_vc = pd.read_csv(os.path.join(BASE_PATH, "noisy_virtual_centers.csv"))
    clean_hyper = pd.read_csv(os.path.join(BASE_PATH, "hypergraph_clean.csv"))
    noisy_hyper = pd.read_csv(os.path.join(BASE_PATH, "hypergraph_noisy.csv"))
    return xi, pi, clean_cp, noisy_cp, clean_vc, noisy_vc, clean_hyper, noisy_hyper


def get_region_polygon(cp_df, hyper_row):
    """Get polygon vertices for a region."""
    min_id = int(hyper_row['min_id'])
    max_id = int(hyper_row['max_id'])
    saddle_ids = eval(hyper_row['boundary_saddles'])

    coords = []
    for cp_id in [min_id] + list(saddle_ids) + [max_id]:
        x = cp_df.iloc[cp_id]['Points_0']
        y = cp_df.iloc[cp_id]['Points_1']
        coords.append([x, y])

    coords = np.array(coords)
    cx, cy = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - cy, coords[:, 0] - cx)
    return coords[np.argsort(angles)]


def get_region_center(cp_df, hyper_row):
    """Get center point of a region for label placement."""
    min_id = int(hyper_row['min_id'])
    max_id = int(hyper_row['max_id'])
    saddle_ids = eval(hyper_row['boundary_saddles'])

    coords = []
    for cp_id in [min_id] + list(saddle_ids) + [max_id]:
        x = cp_df.iloc[cp_id]['Points_0']
        y = cp_df.iloc[cp_id]['Points_1']
        coords.append([x, y])

    return np.mean(coords, axis=0)


def coupling_to_opacity(coupling_value, max_coupling):
    """Convert coupling value to opacity based on confidence."""
    if coupling_value < SPURIOUS_THRESHOLD:
        return 0.0
    normalized = coupling_value / max_coupling
    return OPACITY_MIN + (OPACITY_MAX - OPACITY_MIN) * normalized


def is_spurious(coupling_value):
    """Check if a coupling is below spurious threshold."""
    return coupling_value < SPURIOUS_THRESHOLD


def generate_region_colors(n_regions):
    """Generate distinct colors for regions using a good colormap."""
    if n_regions <= 20:
        cmap = plt.cm.tab20
        colors = [cmap(i / 20) for i in range(n_regions)]
    else:
        cmap1 = plt.cm.tab20
        cmap2 = plt.cm.tab20b
        colors = []
        for i in range(n_regions):
            if i < 20:
                colors.append(cmap1(i / 20))
            else:
                colors.append(cmap2((i - 20) / 20))
    return colors


def visualize_hyperedge_correspondence(save_path=None):
    """Create hyperedge (region) correspondence visualization.

    Shows Clean and Noisy MS complexes side-by-side with:
    - Color = Best match from clean regions
    - Opacity = Coupling confidence
    """

    # Load data
    xi, pi, clean_cp, noisy_cp, clean_vc, noisy_vc, clean_hyper, noisy_hyper = load_data()

    n_clean_regions = len(clean_vc)
    n_noisy_regions = len(noisy_vc)

    # Generate unique colors for clean regions
    clean_colors = generate_region_colors(n_clean_regions)

    # For each noisy region, find best match and coupling strength
    noisy_best_match = np.argmax(xi, axis=0)
    noisy_max_coupling = np.max(xi, axis=0)
    max_xi = xi.max()

    # Count spurious regions
    n_spurious = np.sum(noisy_max_coupling < SPURIOUS_THRESHOLD)

    # Create figure
    fig, ax_spatial = plt.subplots(figsize=(12, 6))

    shift_x = 120  # Shift noisy regions to the right

    # Draw Clean Regions (Left) with ID labels
    for i, row in clean_hyper.iterrows():
        vertices = get_region_polygon(clean_cp, row)
        center = get_region_center(clean_cp, row)
        color = clean_colors[i][:3]

        polygon = Polygon(vertices, closed=True,
                          facecolor=(*color, 0.7), edgecolor='black', linewidth=0.8)
        ax_spatial.add_patch(polygon)

        ax_spatial.text(center[0], center[1], str(i + 1),
                       ha='center', va='center', fontsize=7, fontweight='bold',
                       color='black',
                       path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Draw Noisy Regions (Right, shifted) with inherited colors and matched IDs
    for j, row in noisy_hyper.iterrows():
        vertices = get_region_polygon(noisy_cp, row)
        vertices[:, 0] += shift_x
        center = get_region_center(noisy_cp, row)
        center[0] += shift_x

        best_i = noisy_best_match[j]
        coupling = noisy_max_coupling[j]

        if is_spurious(coupling):
            facecolor = (0.6, 0.6, 0.6, 0.4)
            edgecolor = 'gray'
            polygon = Polygon(vertices, closed=True,
                              facecolor=facecolor, edgecolor=edgecolor, linewidth=0.6)
            ax_spatial.add_patch(polygon)
        else:
            base_color = clean_colors[best_i][:3]
            alpha = coupling_to_opacity(coupling, max_xi)
            facecolor = (*base_color, alpha)
            edgecolor = 'black'

            polygon = Polygon(vertices, closed=True,
                              facecolor=facecolor, edgecolor=edgecolor, linewidth=0.6)
            ax_spatial.add_patch(polygon)

            ax_spatial.text(center[0], center[1], str(best_i + 1),
                           ha='center', va='center', fontsize=6, fontweight='bold',
                           color='black',
                           path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Draw CPs
    for cp_type in [0, 1, 2]:
        mask = clean_cp['CellDimension'] == cp_type
        ax_spatial.scatter(clean_cp.loc[mask, 'Points_0'], clean_cp.loc[mask, 'Points_1'],
                           c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                           s=30, zorder=5, edgecolors='black', linewidths=0.2, alpha=0.7)
        mask = noisy_cp['CellDimension'] == cp_type
        ax_spatial.scatter(noisy_cp.loc[mask, 'Points_0'] + shift_x, noisy_cp.loc[mask, 'Points_1'],
                           c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                           s=30, zorder=5, edgecolors='black', linewidths=0.2, alpha=0.7)

    ax_spatial.axvline(x=110, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

    ax_spatial.text(50, 108, 'Clean MS Complex', ha='center', fontsize=12, fontweight='bold')
    ax_spatial.text(50 + shift_x, 108, 'Noisy MS Complex', ha='center', fontsize=12, fontweight='bold')
    ax_spatial.text(50, -8, f'({n_clean_regions} regions)', ha='center', fontsize=10, style='italic')
    ax_spatial.text(50 + shift_x, -8, f'({n_noisy_regions} regions)', ha='center', fontsize=10, style='italic')

    ax_spatial.set_xlim(-10, 250)
    ax_spatial.set_ylim(-15, 115)
    ax_spatial.set_aspect('equal')
    ax_spatial.axis('off')

    # Title with inline legend
    ax_spatial.set_title('Hyperedge Correspondence: Color = Best Match, Opacity = Confidence',
                         fontsize=12, fontweight='bold', pad=5)

    # Inline CP type legend at bottom of top subfigure
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CP_COLORS[0],
               markersize=7, label='Min', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CP_COLORS[1],
               markersize=7, label='Saddle', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CP_COLORS[2],
               markersize=7, label='Max', markeredgecolor='black', markeredgewidth=0.5),
    ]
    ax_spatial.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9,
                      framealpha=0.9, handletextpad=0.3, columnspacing=0.8,
                      bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('HyperCOT: Hyperedge Correspondence (ξ)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")

    return {
        'n_clean_regions': n_clean_regions,
        'n_noisy_regions': n_noisy_regions,
        'n_spurious': n_spurious,
        'max_coupling': max_xi,
        'mean_coupling': xi.mean(),
        'clean_colors': clean_colors,
        'noisy_best_match': noisy_best_match
    }


def visualize_node_correspondence(save_path=None, coupling_threshold=0.005):
    """Create node correspondence visualization with overlaid CPs.

    Shows geometric correspondence from clean to noisy MS complex:
    - Clean CPs as squares (blue)
    - Noisy CPs as circles (orange)
    - Lines for transport connections where π[i,j] > threshold
    - Line color = coupling strength (darker red = stronger)
    """

    # Load data
    xi, pi, clean_cp, noisy_cp, clean_vc, noisy_vc, clean_hyper, noisy_hyper = load_data()

    n_clean_cps = len(clean_cp)
    n_noisy_cps = len(noisy_cp)

    # Get max coupling for normalization
    max_coupling = pi.max()

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 8))

    # Use OrRd colormap - darker red = stronger coupling
    cmap_coupling = plt.cm.OrRd

    # Collect connections above threshold
    connections = []
    for i in range(n_clean_cps):
        for j in range(n_noisy_cps):
            coupling = pi[i, j]
            if coupling >= coupling_threshold:
                connections.append((i, j, coupling))

    n_connections = len(connections)

    # Sort by coupling (weakest first, so strongest drawn on top)
    connections.sort(key=lambda x: x[2])

    # Draw clean CPs first - solid blue squares
    ax.scatter(clean_cp['Points_0'], clean_cp['Points_1'],
               c='#2166ac', marker='s', s=80, zorder=2,
               edgecolors='white', linewidths=1, alpha=0.9)

    # Draw noisy CPs - hollow orange circles (ring only)
    ax.scatter(noisy_cp['Points_0'], noisy_cp['Points_1'],
               facecolors='none', marker='o', s=120, zorder=3,
               edgecolors='#ff7f0e', linewidths=2.5, alpha=0.95)

    # Draw lines ON TOP of markers so they're always visible
    for i, j, coupling in connections:
        clean_x = clean_cp.iloc[i]['Points_0']
        clean_y = clean_cp.iloc[i]['Points_1']
        noisy_x = noisy_cp.iloc[j]['Points_0']
        noisy_y = noisy_cp.iloc[j]['Points_1']

        # Normalize coupling for color (0.3 to 1 to avoid too light colors)
        norm_coupling = coupling / max_coupling if max_coupling > 0 else 0
        color_val = 0.3 + 0.7 * norm_coupling

        line_color = cmap_coupling(color_val)
        lw = 1.0 + 2.0 * norm_coupling  # Thicker = higher coupling

        ax.plot([clean_x, noisy_x], [clean_y, noisy_y],
                color=line_color, lw=lw, alpha=0.85, zorder=4)

    # Add colorbar for coupling strength
    sm = ScalarMappable(cmap=cmap_coupling, norm=Normalize(0, max_coupling))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Coupling Strength (π)', fontsize=11)

    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_aspect('equal')

    ax.set_title(f'Node Correspondence ({n_connections} connections, threshold={coupling_threshold})\n'
                 f'{n_clean_cps} clean CPs ↔ {n_noisy_cps} noisy CPs',
                 fontsize=13, fontweight='bold')

    # Legend outside figure, above colorbar
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2166ac',
               markersize=10, label='Clean CP', markeredgecolor='white', markeredgewidth=1),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markersize=10, label='Noisy CP', markeredgecolor='#ff7f0e', markeredgewidth=2.5),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
              framealpha=0.95, bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")

    return {
        'n_clean_cps': n_clean_cps,
        'n_noisy_cps': n_noisy_cps,
        'n_connections': n_connections,
        'max_coupling': max_coupling
    }


def main():
    print("=" * 60)
    print("GENERATING HYPERCOT VISUALIZATIONS")
    print("=" * 60)

    # Generate hyperedge correspondence figure
    print("\n1. Generating hyperedge correspondence (ξ)...")
    stats1 = visualize_hyperedge_correspondence(
        os.path.join(BASE_PATH, "hypercot_hyperedge_correspondence.png"))
    print(f"   Clean regions: {stats1['n_clean_regions']}")
    print(f"   Noisy regions: {stats1['n_noisy_regions']}")
    print(f"   Spurious regions: {stats1['n_spurious']}")

    # Generate node correspondence figure
    print("\n2. Generating node correspondence (π)...")
    stats2 = visualize_node_correspondence(
        os.path.join(BASE_PATH, "hypercot_node_correspondence.png"))
    print(f"   Clean CPs: {stats2['n_clean_cps']}")
    print(f"   Noisy CPs: {stats2['n_noisy_cps']}")
    print(f"   Connections: {stats2['n_connections']}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
