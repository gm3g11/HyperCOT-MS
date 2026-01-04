"""
Final HyperCOT Visualization
- Spatial correspondence with numbered lines
- Both ξ (hyperedge) and π (node) matrices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
import os

BASE_PATH = "/Users/gmeng/Desktop/COOT on Morse-Smale"


def load_data():
    """Load all data."""
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


def visualize_final(save_path=None):
    """Create final visualization with both π and ξ."""

    # Load data
    xi, pi, clean_cp, noisy_cp, clean_vc, noisy_vc, clean_hyper, noisy_hyper = load_data()

    n_clean_regions = len(clean_vc)
    n_noisy_regions = len(noisy_vc)
    n_clean_cps = len(clean_cp)
    n_noisy_cps = len(noisy_cp)

    # Colors
    CP_COLORS = {0: '#2166ac', 1: '#4daf4a', 2: '#e41a1c'}
    CP_MARKERS = {0: 'o', 1: 's', 2: '^'}

    # Create figure with better proportions
    fig = plt.figure(figsize=(14, 13))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.6, 1], width_ratios=[1, 1],
                           hspace=0.25, wspace=0.3)

    # ========== TOP: Spatial Correspondence (spans both columns) ==========
    ax_top = fig.add_subplot(gs[0, :])

    shift_x = 120
    np.random.seed(42)
    region_colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # Find best matches
    noisy_to_clean_best = np.argmax(xi, axis=0)

    # Draw clean regions (left)
    clean_patches = []
    for i, row in clean_hyper.iterrows():
        vertices = get_region_polygon(clean_cp, row)
        polygon = Polygon(vertices, closed=True)
        clean_patches.append(polygon)

    clean_collection = PatchCollection(clean_patches, alpha=0.5)
    clean_collection.set_facecolors([region_colors[i % 20] for i in range(n_clean_regions)])
    clean_collection.set_edgecolors('gray')
    clean_collection.set_linewidth(0.8)
    ax_top.add_collection(clean_collection)

    # Draw noisy regions (right, shifted)
    noisy_patches = []
    for i, row in noisy_hyper.iterrows():
        vertices = get_region_polygon(noisy_cp, row)
        vertices[:, 0] += shift_x
        polygon = Polygon(vertices, closed=True)
        noisy_patches.append(polygon)

    noisy_collection = PatchCollection(noisy_patches, alpha=0.5)
    noisy_colors = [region_colors[noisy_to_clean_best[i] % 20] for i in range(n_noisy_regions)]
    noisy_collection.set_facecolors(noisy_colors)
    noisy_collection.set_edgecolors('gray')
    noisy_collection.set_linewidth(0.8)
    ax_top.add_collection(noisy_collection)

    # Virtual centers
    clean_vc_x = clean_vc['vc_x'].values
    clean_vc_y = clean_vc['vc_y'].values
    noisy_vc_x = noisy_vc['vc_x'].values + shift_x
    noisy_vc_y = noisy_vc['vc_y'].values

    ax_top.scatter(clean_vc_x, clean_vc_y, c='white', marker='*', s=80, zorder=6,
                   edgecolors='black', linewidths=0.8)
    ax_top.scatter(noisy_vc_x, noisy_vc_y, c='white', marker='*', s=80, zorder=6,
                   edgecolors='black', linewidths=0.8)

    # Draw top correspondences (no numbers)
    n_top = 20
    max_xi = xi.max()
    flat_xi = xi.flatten()
    top_indices = np.argsort(flat_xi)[-n_top:][::-1]

    for idx in top_indices:
        i, j = np.unravel_index(idx, xi.shape)
        coupling = xi[i, j]
        alpha = min(coupling / max_xi * 1.2, 0.8)
        linewidth = 1 + coupling / max_xi * 3

        # Draw line
        ax_top.plot([clean_vc_x[i], noisy_vc_x[j]],
                    [clean_vc_y[i], noisy_vc_y[j]],
                    color=region_colors[i % 20],
                    alpha=alpha, linewidth=linewidth, zorder=4)

    # CPs
    for cp_type in [0, 1, 2]:
        mask = clean_cp['CellDimension'] == cp_type
        ax_top.scatter(clean_cp.loc[mask, 'Points_0'], clean_cp.loc[mask, 'Points_1'],
                       c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                       s=35, zorder=5, edgecolors='black', linewidths=0.3, alpha=0.8)
        mask = noisy_cp['CellDimension'] == cp_type
        ax_top.scatter(noisy_cp.loc[mask, 'Points_0'] + shift_x, noisy_cp.loc[mask, 'Points_1'],
                       c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                       s=35, zorder=5, edgecolors='black', linewidths=0.3, alpha=0.8)

    # Dividing line
    ax_top.axvline(x=110, color='black', linestyle='--', alpha=0.4, linewidth=1.5)

    # Labels
    ax_top.text(50, 107, 'Clean MS Complex', ha='center', fontsize=12, fontweight='bold')
    ax_top.text(50 + shift_x, 107, 'Noisy MS Complex', ha='center', fontsize=12, fontweight='bold')
    ax_top.text(50, -8, f'({n_clean_regions} regions, {n_clean_cps} CPs)',
                ha='center', fontsize=9, style='italic')
    ax_top.text(50 + shift_x, -8, f'({n_noisy_regions} regions, {n_noisy_cps} CPs)',
                ha='center', fontsize=9, style='italic')

    ax_top.set_xlim(-5, 245)
    ax_top.set_ylim(-15, 115)
    ax_top.set_aspect('equal')
    ax_top.axis('off')
    ax_top.set_title('Spatial Hyperedge Correspondence',
                     fontsize=12, fontweight='bold', pad=15)

    # Legend - centered below spatial view
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CP_COLORS[0],
               markersize=8, label='Min', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CP_COLORS[1],
               markersize=8, label='Saddle', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CP_COLORS[2],
               markersize=8, label='Max', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white',
               markersize=11, label='Virtual Center', markeredgecolor='black', markeredgewidth=0.8),
    ]
    ax_top.legend(handles=legend_elements, loc='upper center', fontsize=9,
                  framealpha=0.95, ncol=4, bbox_to_anchor=(0.5, -0.06),
                  handletextpad=0.3, columnspacing=1.5)

    # ========== BOTTOM LEFT: ξ Matrix (Hyperedge Coupling) ==========
    ax_xi = fig.add_subplot(gs[1, 0])

    im_xi = ax_xi.imshow(xi, aspect='auto', cmap='YlOrRd')
    cbar_xi = plt.colorbar(im_xi, ax=ax_xi, shrink=0.85, pad=0.02)
    cbar_xi.set_label('Coupling ξ', fontsize=10)

    ax_xi.set_xlabel('Noisy Region Index', fontsize=10)
    ax_xi.set_ylabel('Clean Region Index', fontsize=10)

    # Tick labels
    ax_xi.set_xticks(np.arange(0, n_noisy_regions, 10))
    ax_xi.set_xticklabels([str(i) for i in range(0, n_noisy_regions, 10)], fontsize=8)
    ax_xi.set_yticks(np.arange(0, n_clean_regions, 5))
    ax_xi.set_yticklabels([str(i) for i in range(0, n_clean_regions, 5)], fontsize=8)

    ax_xi.set_title(f'ξ Matrix (Hyperedge Coupling)',
                    fontsize=11, fontweight='bold', pad=5)

    # ========== BOTTOM RIGHT: π Matrix (Node Coupling) ==========
    ax_pi = fig.add_subplot(gs[1, 1])

    im_pi = ax_pi.imshow(pi, aspect='auto', cmap='Blues')
    cbar_pi = plt.colorbar(im_pi, ax=ax_pi, shrink=0.85, pad=0.02)
    cbar_pi.set_label('Coupling π', fontsize=10)

    ax_pi.set_xlabel('Noisy CP Index', fontsize=10)
    ax_pi.set_ylabel('Clean CP Index', fontsize=10)

    # Tick labels
    ax_pi.set_xticks(np.arange(0, n_noisy_cps, 10))
    ax_pi.set_xticklabels([str(i) for i in range(0, n_noisy_cps, 10)], fontsize=8)
    ax_pi.set_yticks(np.arange(0, n_clean_cps, 10))
    ax_pi.set_yticklabels([str(i) for i in range(0, n_clean_cps, 10)], fontsize=8)

    ax_pi.set_title(f'π Matrix (Node Coupling)',
                    fontsize=11, fontweight='bold', pad=5)

    # Main title
    fig.suptitle('HyperCOT: Clean ↔ Noisy MS Complex Correspondence',
                 fontsize=13, fontweight='bold', y=0.99)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")


def main():
    print("Generating final HyperCOT visualization...")
    save_path = os.path.join(BASE_PATH, "hypercot_detailed_correspondence.png")
    visualize_final(save_path=save_path)
    print("Done!")


if __name__ == "__main__":
    main()
