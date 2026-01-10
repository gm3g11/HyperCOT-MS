"""
HyperCOT Visualization with Probability-Based Region Coloring

2x2 figure layout:
- Top Left: Clean MS complex (regions colored by ID)
- Top Right: Noisy MS complex (color = best match, opacity = coupling strength)
- Bottom Left: ξ coupling matrix (region coupling)
- Bottom Right: π coupling matrix (node coupling)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
import os

BASE_PATH = "/Users/gmeng/Desktop/COOT on Morse-Smale"

# Thresholds and parameters
# Note: SPURIOUS_THRESHOLD is relative to max coupling, set dynamically
SPURIOUS_RATIO = 0.3          # Below 30% of max coupling, consider spurious
OPACITY_MIN = 0.35            # Minimum opacity for matched regions
OPACITY_MAX = 0.85            # Maximum opacity for strong matches
TOP_K_LINES = 15              # Number of top correspondence lines to draw
LINE_WIDTH_MIN = 1.0
LINE_WIDTH_MAX = 4.0

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


def coupling_to_opacity(coupling_value, max_coupling):
    """Convert coupling value to opacity."""
    spurious_threshold = max_coupling * SPURIOUS_RATIO
    if coupling_value < spurious_threshold:
        return 0.3  # Low opacity for spurious
    # Scale to [OPACITY_MIN, OPACITY_MAX]
    normalized = coupling_value / max_coupling
    return OPACITY_MIN + (OPACITY_MAX - OPACITY_MIN) * normalized


def is_spurious(coupling_value, max_coupling):
    """Check if a coupling is below spurious threshold."""
    return coupling_value < max_coupling * SPURIOUS_RATIO


def visualize_hypercot(save_path=None):
    """Create HyperCOT visualization matching the reference format:
    - Top: Single panel with Clean and Noisy side-by-side with correspondence lines
    - Bottom: ξ matrix (left) and π matrix (right) with linear scale
    """

    # Load data
    xi, pi, clean_cp, noisy_cp, clean_vc, noisy_vc, clean_hyper, noisy_hyper = load_data()

    n_clean_regions = len(clean_vc)
    n_noisy_regions = len(noisy_vc)
    n_clean_cps = len(clean_cp)
    n_noisy_cps = len(noisy_cp)

    # Generate consistent colors for clean regions
    np.random.seed(42)
    region_cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    clean_region_colors = {i: region_cmap[i % 20] for i in range(n_clean_regions)}

    # For each noisy region, find best match and coupling strength
    noisy_best_match = np.argmax(xi, axis=0)  # Best clean region for each noisy
    noisy_max_coupling = np.max(xi, axis=0)   # Coupling strength
    max_xi = xi.max()

    # Count spurious regions
    spurious_threshold = max_xi * SPURIOUS_RATIO
    n_spurious = np.sum(noisy_max_coupling < spurious_threshold)

    # Create figure with gridspec: top panel wide, bottom two panels
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.25, wspace=0.25)

    # =========================================================================
    # Top: Spatial Hyperedge Correspondence (Clean and Noisy side-by-side)
    # =========================================================================
    ax_spatial = fig.add_subplot(gs[0, :])  # Span both columns

    shift_x = 120  # Shift noisy regions to the right

    # Draw Clean Regions (Left)
    for i, row in clean_hyper.iterrows():
        vertices = get_region_polygon(clean_cp, row)
        color = clean_region_colors[i][:3]
        polygon = Polygon(vertices, closed=True,
                          facecolor=(*color, 0.6), edgecolor='gray', linewidth=0.8)
        ax_spatial.add_patch(polygon)

    # Draw Noisy Regions (Right, shifted) with matching colors
    for i, row in noisy_hyper.iterrows():
        vertices = get_region_polygon(noisy_cp, row)
        vertices[:, 0] += shift_x  # Shift right
        best_r = noisy_best_match[i]
        coupling = noisy_max_coupling[i]

        if is_spurious(coupling, max_xi):
            facecolor = (0.7, 0.7, 0.7, 0.5)
            edgecolor = 'darkgray'
        else:
            base_color = clean_region_colors[best_r][:3]
            alpha = coupling_to_opacity(coupling, max_xi)
            facecolor = (*base_color, alpha)
            edgecolor = 'gray'

        polygon = Polygon(vertices, closed=True,
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8)
        ax_spatial.add_patch(polygon)

    # Draw Virtual Centers
    ax_spatial.scatter(clean_vc['vc_x'], clean_vc['vc_y'],
                       c='white', marker='*', s=100, zorder=6,
                       edgecolors='black', linewidths=0.8, label='Virtual Center')
    ax_spatial.scatter(noisy_vc['vc_x'] + shift_x, noisy_vc['vc_y'],
                       c='white', marker='*', s=100, zorder=6,
                       edgecolors='black', linewidths=0.8)

    # Draw CPs
    for cp_type in [0, 1, 2]:
        # Clean CPs
        mask = clean_cp['CellDimension'] == cp_type
        ax_spatial.scatter(clean_cp.loc[mask, 'Points_0'], clean_cp.loc[mask, 'Points_1'],
                           c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                           s=60, zorder=5, edgecolors='black', linewidths=0.3, alpha=0.9)
        # Noisy CPs (shifted)
        mask = noisy_cp['CellDimension'] == cp_type
        ax_spatial.scatter(noisy_cp.loc[mask, 'Points_0'] + shift_x, noisy_cp.loc[mask, 'Points_1'],
                           c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                           s=60, zorder=5, edgecolors='black', linewidths=0.3, alpha=0.9)

    # Draw Top-K Correspondence Lines between VCs
    flat_xi = xi.flatten()
    top_indices = np.argsort(flat_xi)[-TOP_K_LINES:][::-1]

    for idx in top_indices:
        i, j = np.unravel_index(idx, xi.shape)
        coupling = xi[i, j]

        # Line properties based on coupling
        alpha = 0.3 + 0.6 * (coupling / max_xi)
        linewidth = LINE_WIDTH_MIN + (LINE_WIDTH_MAX - LINE_WIDTH_MIN) * (coupling / max_xi)

        ax_spatial.plot([clean_vc.iloc[i]['vc_x'], noisy_vc.iloc[j]['vc_x'] + shift_x],
                        [clean_vc.iloc[i]['vc_y'], noisy_vc.iloc[j]['vc_y']],
                        color=clean_region_colors[i][:3],
                        alpha=alpha, linewidth=linewidth, zorder=3)

    # Dividing line
    ax_spatial.axvline(x=110, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

    # Labels
    ax_spatial.text(50, 108, 'Clean MS Complex', ha='center', fontsize=12, fontweight='bold')
    ax_spatial.text(50 + shift_x, 108, 'Noisy MS Complex', ha='center', fontsize=12, fontweight='bold')
    ax_spatial.text(50, -8, f'({n_clean_regions} regions, {n_clean_cps} CPs)', ha='center', fontsize=10, style='italic')
    ax_spatial.text(50 + shift_x, -8, f'({n_noisy_regions} regions, {n_noisy_cps} CPs)', ha='center', fontsize=10, style='italic')

    ax_spatial.set_xlim(-10, 250)
    ax_spatial.set_ylim(-15, 115)
    ax_spatial.set_aspect('equal')
    ax_spatial.axis('off')
    ax_spatial.set_title('Spatial Hyperedge Correspondence', fontsize=13, fontweight='bold', pad=10)

    # Legend for spatial plot
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CP_COLORS[0],
               markersize=8, label='Min', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CP_COLORS[1],
               markersize=8, label='Saddle', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CP_COLORS[2],
               markersize=8, label='Max', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white',
               markersize=10, label='Virtual Center', markeredgecolor='black', markeredgewidth=0.8),
    ]
    ax_spatial.legend(handles=legend_elements, loc='upper center', fontsize=9,
                      framealpha=0.95, ncol=4, bbox_to_anchor=(0.5, -0.02))

    # =========================================================================
    # Bottom Left: ξ Matrix (Hyperedge Coupling) - LINEAR scale like reference
    # =========================================================================
    ax_xi = fig.add_subplot(gs[1, 0])

    im_xi = ax_xi.imshow(xi, aspect='auto', cmap='Oranges')
    cbar_xi = plt.colorbar(im_xi, ax=ax_xi, shrink=0.85, pad=0.02)
    cbar_xi.set_label('Coupling ξ', fontsize=10)

    ax_xi.set_xlabel('Noisy Region Index', fontsize=10)
    ax_xi.set_ylabel('Clean Region Index', fontsize=10)
    ax_xi.set_title('ξ Matrix (Hyperedge Coupling)', fontsize=11, fontweight='bold')

    # =========================================================================
    # Bottom Right: π Matrix (Node Coupling) - Sorted by type with colored blocks
    # =========================================================================
    ax_pi = fig.add_subplot(gs[1, 1])

    # Count CPs by type
    CP_NAMES = {0: 'Min', 1: 'Sad', 2: 'Max'}
    clean_counts = {t: sum(clean_cp['CellDimension'] == t) for t in [0, 1, 2]}
    noisy_counts = {t: sum(noisy_cp['CellDimension'] == t) for t in [0, 1, 2]}

    # Get indices sorted by type (Min=0, Saddle=1, Max=2)
    clean_order = np.argsort(clean_cp['CellDimension'].values)
    noisy_order = np.argsort(noisy_cp['CellDimension'].values)

    # Reorder coupling matrix by type
    pi_sorted = pi[clean_order][:, noisy_order]

    # Type boundaries
    clean_boundaries = [0, clean_counts[0], clean_counts[0] + clean_counts[1], n_clean_cps]
    noisy_boundaries = [0, noisy_counts[0], noisy_counts[0] + noisy_counts[1], n_noisy_cps]

    # Add colored background regions for type matching (draw first)
    from matplotlib.patches import Rectangle
    for i, (cb1, cb2) in enumerate(zip(clean_boundaries[:-1], clean_boundaries[1:])):
        for j, (nb1, nb2) in enumerate(zip(noisy_boundaries[:-1], noisy_boundaries[1:])):
            color = '#c8e6c9' if i == j else '#ffcdd2'  # green if same type, pink otherwise
            rect = Rectangle((nb1 - 0.5, cb1 - 0.5), nb2 - nb1, cb2 - cb1,
                            facecolor=color, edgecolor='black', linewidth=1.0, zorder=0)
            ax_pi.add_patch(rect)

    # Plot the sorted coupling matrix with transparent low values to show blocks
    from matplotlib.colors import LinearSegmentedColormap
    # Create custom colormap: transparent white -> red
    colors = [(1, 1, 1, 0), (1, 0.8, 0.8, 0.5), (1, 0.4, 0.4, 0.8), (0.7, 0, 0, 1)]
    cmap_transparent = LinearSegmentedColormap.from_list('TransparentReds', colors)
    im_pi = ax_pi.imshow(pi_sorted, aspect='auto', cmap=cmap_transparent, zorder=1)
    cbar_pi = plt.colorbar(im_pi, ax=ax_pi, shrink=0.85, pad=0.02)
    cbar_pi.set_label('Coupling π', fontsize=10)

    # Type labels on axes
    for i, (b1, b2) in enumerate(zip(clean_boundaries[:-1], clean_boundaries[1:])):
        mid = (b1 + b2) / 2
        ax_pi.text(-2.5, mid, CP_NAMES[i], ha='right', va='center', fontsize=9,
                   color=CP_COLORS[i], fontweight='bold')

    for j, (b1, b2) in enumerate(zip(noisy_boundaries[:-1], noisy_boundaries[1:])):
        mid = (b1 + b2) / 2
        ax_pi.text(mid, n_clean_cps + 1.5, CP_NAMES[j], ha='center', va='top', fontsize=9,
                   color=CP_COLORS[j], fontweight='bold')

    ax_pi.set_xlabel('Noisy CP Index (sorted by type)', fontsize=10)
    ax_pi.set_ylabel('Clean CP Index (sorted by type)', fontsize=10)
    ax_pi.set_title('π Matrix (Node Coupling)', fontsize=11, fontweight='bold')
    ax_pi.set_xlim(-0.5, n_noisy_cps - 0.5)
    ax_pi.set_ylim(n_clean_cps - 0.5, -0.5)

    # Main title
    fig.suptitle('HyperCOT: Clean ↔ Noisy MS Complex Correspondence',
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")

    return {
        'n_clean_regions': n_clean_regions,
        'n_noisy_regions': n_noisy_regions,
        'n_spurious': n_spurious,
        'max_coupling': max_xi,
        'mean_coupling': xi.mean()
    }


def visualize_correspondence_lines(save_path=None):
    """Create visualization with correspondence lines between regions."""

    # Load data
    xi, pi, clean_cp, noisy_cp, clean_vc, noisy_vc, clean_hyper, noisy_hyper = load_data()

    n_clean_regions = len(clean_vc)
    n_noisy_regions = len(noisy_vc)

    # Generate colors
    np.random.seed(42)
    region_cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    clean_region_colors = {i: region_cmap[i % 20] for i in range(n_clean_regions)}

    # Find best matches
    noisy_best_match = np.argmax(xi, axis=0)
    noisy_max_coupling = np.max(xi, axis=0)
    max_xi = xi.max()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    shift_x = 120  # Shift noisy regions to the right

    # =========================================================================
    # Draw Clean Regions (Left)
    # =========================================================================
    clean_patches = []
    for i, row in clean_hyper.iterrows():
        vertices = get_region_polygon(clean_cp, row)
        polygon = Polygon(vertices, closed=True)
        clean_patches.append(polygon)

    clean_collection = PatchCollection(clean_patches, alpha=0.5)
    clean_collection.set_facecolors([clean_region_colors[i][:3] for i in range(n_clean_regions)])
    clean_collection.set_edgecolors('gray')
    clean_collection.set_linewidth(0.8)
    ax.add_collection(clean_collection)

    # =========================================================================
    # Draw Noisy Regions (Right, shifted)
    # =========================================================================
    for i, row in noisy_hyper.iterrows():
        vertices = get_region_polygon(noisy_cp, row)
        vertices[:, 0] += shift_x
        best_r = noisy_best_match[i]
        coupling = noisy_max_coupling[i]

        if is_spurious(coupling, max_xi):
            facecolor = (0.7, 0.7, 0.7, 0.4)
            edgecolor = 'darkgray'
        else:
            base_color = clean_region_colors[best_r][:3]
            alpha = coupling_to_opacity(coupling, max_xi)
            facecolor = (*base_color, alpha)
            edgecolor = 'gray'

        polygon = Polygon(vertices, closed=True,
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8)
        ax.add_patch(polygon)

    # =========================================================================
    # Draw Virtual Centers
    # =========================================================================
    clean_vc_x = clean_vc['vc_x'].values
    clean_vc_y = clean_vc['vc_y'].values
    noisy_vc_x = noisy_vc['vc_x'].values + shift_x
    noisy_vc_y = noisy_vc['vc_y'].values

    ax.scatter(clean_vc_x, clean_vc_y, c='white', marker='*', s=80, zorder=6,
               edgecolors='black', linewidths=0.8)
    ax.scatter(noisy_vc_x, noisy_vc_y, c='white', marker='*', s=80, zorder=6,
               edgecolors='black', linewidths=0.8)

    # =========================================================================
    # Draw Top-K Correspondence Lines
    # =========================================================================
    flat_xi = xi.flatten()
    top_indices = np.argsort(flat_xi)[-TOP_K_LINES:][::-1]

    for idx in top_indices:
        i, j = np.unravel_index(idx, xi.shape)
        coupling = xi[i, j]

        # Line properties based on coupling
        alpha = 0.4 + 0.5 * (coupling / max_xi)
        linewidth = LINE_WIDTH_MIN + (LINE_WIDTH_MAX - LINE_WIDTH_MIN) * (coupling / max_xi)

        ax.plot([clean_vc_x[i], noisy_vc_x[j]],
                [clean_vc_y[i], noisy_vc_y[j]],
                color=clean_region_colors[i][:3],
                alpha=alpha, linewidth=linewidth, zorder=4)

    # =========================================================================
    # Draw CPs
    # =========================================================================
    for cp_type in [0, 1, 2]:
        mask = clean_cp['CellDimension'] == cp_type
        ax.scatter(clean_cp.loc[mask, 'Points_0'], clean_cp.loc[mask, 'Points_1'],
                   c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                   s=35, zorder=5, edgecolors='black', linewidths=0.3, alpha=0.8)
        mask = noisy_cp['CellDimension'] == cp_type
        ax.scatter(noisy_cp.loc[mask, 'Points_0'] + shift_x, noisy_cp.loc[mask, 'Points_1'],
                   c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                   s=35, zorder=5, edgecolors='black', linewidths=0.3, alpha=0.8)

    # Dividing line
    ax.axvline(x=110, color='black', linestyle='--', alpha=0.4, linewidth=1.5)

    # Labels
    ax.text(50, 110, 'Clean MS Complex', ha='center', fontsize=12, fontweight='bold')
    ax.text(50 + shift_x, 110, 'Noisy MS Complex', ha='center', fontsize=12, fontweight='bold')
    ax.text(50, -10, f'({n_clean_regions} regions)', ha='center', fontsize=9, style='italic')
    ax.text(50 + shift_x, -10, f'({n_noisy_regions} regions)', ha='center', fontsize=9, style='italic')

    ax.set_xlim(-10, 250)
    ax.set_ylim(-18, 118)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CP_COLORS[0],
               markersize=8, label='Min', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CP_COLORS[1],
               markersize=8, label='Saddle', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CP_COLORS[2],
               markersize=8, label='Max', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white',
               markersize=10, label='VC', markeredgecolor='black', markeredgewidth=0.8),
        Line2D([0], [0], color='gray', linewidth=2, label=f'Top-{TOP_K_LINES} correspondences'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=9,
              framealpha=0.95, ncol=5, bbox_to_anchor=(0.5, -0.02))

    # Title
    ax.set_title('HyperCOT: Spatial Region Correspondence\n(Line width = coupling strength, Color = clean region ID)',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("GENERATING HYPERCOT VISUALIZATIONS")
    print("=" * 60)

    # Generate 2x2 visualization
    print("\n1. Generating 2x2 visualization...")
    stats = visualize_hypercot(os.path.join(BASE_PATH, "hypercot_detailed_correspondence.png"))
    print(f"   Clean regions: {stats['n_clean_regions']}")
    print(f"   Noisy regions: {stats['n_noisy_regions']}")
    print(f"   Spurious regions: {stats['n_spurious']}")
    print(f"   Max coupling: {stats['max_coupling']:.6f}")

    # Generate correspondence lines visualization
    print("\n2. Generating correspondence lines visualization...")
    visualize_correspondence_lines(os.path.join(BASE_PATH, "hypercot_spatial_correspondence.png"))

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
