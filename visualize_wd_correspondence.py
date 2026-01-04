"""
Visualize Wasserstein Distance (WD) Correspondence between Clean and Noisy MS Complexes

WD uses only scalar field values (data column) for matching - no structural information.
Format matches gwd_point_edge_correspondence.png style.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import ot
import os

BASE_PATH = "/Users/gmeng/Desktop/COOT on Morse-Smale"


def load_data():
    """Load critical points data."""
    clean_cp = pd.read_csv(os.path.join(BASE_PATH, "clean_critical_points.csv"))
    noisy_cp = pd.read_csv(os.path.join(BASE_PATH, "noisy_critical_points.csv"))
    return clean_cp, noisy_cp


def compute_wasserstein(clean_cp, noisy_cp):
    """
    Compute Wasserstein distance using scalar field values.
    Cost C[i,j] = |f_i - f_j| (absolute difference of scalar values)
    """
    scalar1 = clean_cp['data'].values
    scalar2 = noisy_cp['data'].values

    n1, n2 = len(scalar1), len(scalar2)

    # Cost matrix: absolute difference of scalar values
    C = np.abs(scalar1[:, None] - scalar2[None, :])

    # Uniform measures
    mu = np.ones(n1) / n1
    nu = np.ones(n2) / n2

    # Solve optimal transport
    coupling = ot.emd(mu, nu, C)
    distance = np.sum(coupling * C)

    return distance, coupling, C


def extract_correspondences(coupling, clean_cp, noisy_cp, n_top=40):
    """Extract top correspondences from coupling matrix."""
    n1, n2 = coupling.shape

    correspondences = []
    for i in range(n1):
        for j in range(n2):
            if coupling[i, j] > 1e-10:
                correspondences.append({
                    'clean_idx': i,
                    'noisy_idx': j,
                    'weight': coupling[i, j],
                    'clean_type': clean_cp.iloc[i]['CellDimension'],
                    'noisy_type': noisy_cp.iloc[j]['CellDimension'],
                    'clean_scalar': clean_cp.iloc[i]['data'],
                    'noisy_scalar': noisy_cp.iloc[j]['data'],
                })

    correspondences.sort(key=lambda x: x['weight'], reverse=True)
    return correspondences[:n_top]


def compute_type_preservation(correspondences):
    """Compute percentage of type-preserving matches."""
    if not correspondences:
        return 0.0
    type_match = sum(1 for c in correspondences if c['clean_type'] == c['noisy_type'])
    return type_match / len(correspondences) * 100


def visualize_wd_correspondence(clean_cp, noisy_cp, coupling, wd_dist, correspondences, save_path):
    """Create WD correspondence visualization matching GWD style."""

    # Colors and markers for CP types
    CP_COLORS = {0: '#2166ac', 1: '#4daf4a', 2: '#e41a1c'}
    CP_NAMES = {0: 'Min', 1: 'Sad', 2: 'Max'}

    n1, n2 = len(clean_cp), len(noisy_cp)

    # Count CPs by type
    clean_counts = {t: sum(clean_cp['CellDimension'] == t) for t in [0, 1, 2]}
    noisy_counts = {t: sum(noisy_cp['CellDimension'] == t) for t in [0, 1, 2]}

    # Get indices sorted by type (Min=0, Saddle=1, Max=2)
    clean_order = np.argsort(clean_cp['CellDimension'].values)
    noisy_order = np.argsort(noisy_cp['CellDimension'].values)

    # Reorder coupling matrix by type
    coupling_sorted = coupling[clean_order][:, noisy_order]

    # Type boundaries for clean (sorted order: Min, Saddle, Max)
    clean_boundaries = [0, clean_counts[0], clean_counts[0] + clean_counts[1], n1]
    noisy_boundaries = [0, noisy_counts[0], noisy_counts[0] + noisy_counts[1], n2]

    # Create figure
    fig = plt.figure(figsize=(12, 5.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # ========== LEFT: Coupling Matrix with Type Regions ==========
    ax1 = fig.add_subplot(gs[0])

    # Plot coupling matrix with log scale
    coupling_plot = coupling_sorted.copy()
    coupling_plot[coupling_plot < 1e-10] = np.nan

    im = ax1.imshow(coupling_plot, aspect='auto', cmap='YlOrRd',
                    norm=LogNorm(vmin=1e-3, vmax=coupling_plot[~np.isnan(coupling_plot)].max()))

    # Add colored background regions for type matching
    # Green for diagonal (type-matching), pink for off-diagonal
    for i, (cb1, cb2) in enumerate(zip(clean_boundaries[:-1], clean_boundaries[1:])):
        for j, (nb1, nb2) in enumerate(zip(noisy_boundaries[:-1], noisy_boundaries[1:])):
            color = '#c8e6c9' if i == j else '#ffcdd2'  # green if same type, pink otherwise
            rect = Rectangle((nb1 - 0.5, cb1 - 0.5), nb2 - nb1, cb2 - cb1,
                            facecolor=color, edgecolor='black', linewidth=1, zorder=0)
            ax1.add_patch(rect)

    # Re-plot the coupling on top
    ax1.imshow(coupling_plot, aspect='auto', cmap='YlOrRd',
               norm=LogNorm(vmin=1e-3, vmax=coupling_plot[~np.isnan(coupling_plot)].max()))

    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
    cbar.set_label('Transport Mass (log scale)', fontsize=9)

    # Type labels on axes
    for i, (b1, b2) in enumerate(zip(clean_boundaries[:-1], clean_boundaries[1:])):
        mid = (b1 + b2) / 2
        ax1.text(-3, mid, CP_NAMES[i], ha='right', va='center', fontsize=10,
                color=CP_COLORS[i], fontweight='bold')

    for j, (b1, b2) in enumerate(zip(noisy_boundaries[:-1], noisy_boundaries[1:])):
        mid = (b1 + b2) / 2
        ax1.text(mid, n1 + 2, CP_NAMES[j], ha='center', va='top', fontsize=10,
                color=CP_COLORS[j], fontweight='bold')

    ax1.set_xlabel('Noisy CP Index', fontsize=10)
    ax1.set_ylabel('Clean CP Index', fontsize=10)
    ax1.set_title('WD Coupling Matrix', fontsize=11, fontweight='bold')

    # ========== MIDDLE: Bipartite CP Correspondences ==========
    ax2 = fig.add_subplot(gs[1])

    # Layout: CPs grouped by type vertically
    # Clean on left, Noisy on right
    left_x, right_x = 0.15, 0.85

    # Compute y positions for each CP grouped by type
    def get_y_positions(counts, n_total):
        """Get y positions grouped by type (Max at top, Min at bottom)."""
        positions = {}
        y_ranges = {
            2: (0.75, 0.95),  # Max at top
            1: (0.35, 0.65),  # Saddle in middle
            0: (0.05, 0.25),  # Min at bottom
        }
        idx = 0
        for cp_type in [0, 1, 2]:
            count = counts[cp_type]
            if count > 0:
                y_min, y_max = y_ranges[cp_type]
                ys = np.linspace(y_max, y_min, count)
                for i, y in enumerate(ys):
                    positions[idx] = (cp_type, y)
                    idx += 1
        return positions

    # Build position mappings
    clean_positions = {}
    noisy_positions = {}

    # For clean CPs (sorted by type)
    idx = 0
    for cp_type in [0, 1, 2]:
        mask = clean_cp['CellDimension'] == cp_type
        count = mask.sum()
        y_ranges = {2: (0.75, 0.95), 1: (0.35, 0.65), 0: (0.05, 0.25)}
        if count > 0:
            y_min, y_max = y_ranges[cp_type]
            ys = np.linspace(y_max, y_min, count)
            orig_indices = clean_cp.index[mask].tolist()
            for orig_idx, y in zip(orig_indices, ys):
                clean_positions[orig_idx] = (cp_type, y)

    # For noisy CPs (sorted by type)
    for cp_type in [0, 1, 2]:
        mask = noisy_cp['CellDimension'] == cp_type
        count = mask.sum()
        y_ranges = {2: (0.75, 0.95), 1: (0.35, 0.65), 0: (0.05, 0.25)}
        if count > 0:
            y_min, y_max = y_ranges[cp_type]
            ys = np.linspace(y_max, y_min, count)
            orig_indices = noisy_cp.index[mask].tolist()
            for orig_idx, y in zip(orig_indices, ys):
                noisy_positions[orig_idx] = (cp_type, y)

    # Draw CPs
    for idx, (cp_type, y) in clean_positions.items():
        ax2.scatter(left_x, y, c=CP_COLORS[cp_type], s=50, zorder=5,
                   edgecolors='black', linewidths=0.5,
                   marker='o' if cp_type == 0 else ('s' if cp_type == 1 else '^'))

    for idx, (cp_type, y) in noisy_positions.items():
        ax2.scatter(right_x, y, c=CP_COLORS[cp_type], s=50, zorder=5,
                   edgecolors='black', linewidths=0.5,
                   marker='o' if cp_type == 0 else ('s' if cp_type == 1 else '^'))

    # Draw correspondence lines
    max_weight = max(c['weight'] for c in correspondences) if correspondences else 1

    for corr in correspondences:
        clean_idx = corr['clean_idx']
        noisy_idx = corr['noisy_idx']

        if clean_idx in clean_positions and noisy_idx in noisy_positions:
            clean_type, clean_y = clean_positions[clean_idx]
            noisy_type, noisy_y = noisy_positions[noisy_idx]

            # Green for type match, red for mismatch
            color = '#4daf4a' if clean_type == noisy_type else '#e41a1c'
            alpha = min(corr['weight'] / max_weight * 1.5, 0.8)
            linewidth = 0.5 + corr['weight'] / max_weight * 2

            ax2.plot([left_x, right_x], [clean_y, noisy_y],
                    color=color, alpha=alpha, linewidth=linewidth, zorder=2)

    # Type labels
    for cp_type, (y_min, y_max) in [(2, (0.75, 0.95)), (1, (0.35, 0.65)), (0, (0.05, 0.25))]:
        mid_y = (y_min + y_max) / 2
        ax2.text(left_x - 0.08, mid_y, CP_NAMES[cp_type], ha='right', va='center',
                fontsize=10, color=CP_COLORS[cp_type], fontweight='bold')
        ax2.text(right_x + 0.08, mid_y, CP_NAMES[cp_type], ha='left', va='center',
                fontsize=10, color=CP_COLORS[cp_type], fontweight='bold')

    # Dashed lines between type groups
    for y in [0.30, 0.70]:
        ax2.axhline(y=y, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    ax2.text(left_x, -0.02, f'Clean ({n1})', ha='center', va='top', fontsize=10, fontweight='bold')
    ax2.text(right_x, -0.02, f'Noisy ({n2})', ha='center', va='top', fontsize=10, fontweight='bold')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.05, 1.0)
    ax2.axis('off')
    ax2.set_title(f'CP Correspondences (Top {len(correspondences)})', fontsize=11, fontweight='bold')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CP_COLORS[0],
               markersize=7, label='Min', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CP_COLORS[1],
               markersize=7, label='Sad', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CP_COLORS[2],
               markersize=7, label='Max', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], color='#4daf4a', linewidth=2, label='Match'),
        Line2D([0], [0], color='#e41a1c', linewidth=2, label='Mismatch'),
    ]
    ax2.legend(handles=legend_elements, loc='lower center', fontsize=8,
              ncol=5, bbox_to_anchor=(0.5, -0.08), framealpha=0.9)

    # ========== Bottom Summary Bar ==========
    type_pres = compute_type_preservation(correspondences)

    summary_text = (f"WD = {wd_dist:.4f}  |  "
                   f"Clean: {n1} CPs  |  Noisy: {n2} CPs  |  "
                   f"Type-preserving: {type_pres:.1f}%  |  "
                   f"Cost: |f_i - f_j| (scalar difference)")

    fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#ffffcc', edgecolor='#cccc00', linewidth=1))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Save PNG
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")

    # Save PDF
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")

    plt.close()


def main():
    print("=" * 60)
    print("WASSERSTEIN DISTANCE VISUALIZATION")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    clean_cp, noisy_cp = load_data()
    print(f"  Clean CPs: {len(clean_cp)}")
    print(f"  Noisy CPs: {len(noisy_cp)}")

    # Compute WD
    print("\nComputing Wasserstein Distance...")
    print("  Cost: C[i,j] = |f_i - f_j| (scalar field difference)")
    wd_dist, coupling, cost_matrix = compute_wasserstein(clean_cp, noisy_cp)
    print(f"  WD = {wd_dist:.6f}")

    # Extract correspondences
    print("\nExtracting correspondences...")
    correspondences = extract_correspondences(coupling, clean_cp, noisy_cp, n_top=40)
    type_pres = compute_type_preservation(correspondences)
    print(f"  Type preservation: {type_pres:.1f}%")

    # Visualize
    print("\nGenerating visualization...")
    save_path = os.path.join(BASE_PATH, "wd_correspondence_refined.png")
    visualize_wd_correspondence(clean_cp, noisy_cp, coupling, wd_dist, correspondences, save_path)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
