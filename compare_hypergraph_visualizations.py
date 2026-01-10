"""
Compare different visualization approaches for hypergraph correspondence.

Implements 7 approaches:
A. Row-wise Normalization - Normalize each row to show relative best match
B. Sankey/Alluvial Diagram - Flow diagram showing correspondence strength
C. Top-K Lines Only - Show only the strongest K correspondence lines
D. Hungarian Assignment - One-to-one optimal matching with confidence
E. Bipartite Graph Layout - Parallel axes with connecting lines
F. Threshold-based - Show only correspondences above threshold
G. Clustered Matching - Group similar regions by matching patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle, ConnectionPatch
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.optimize import linear_sum_assignment
import matplotlib.cm as cm
import os

BASE_PATH = "/Users/gmeng/Desktop/COOT on Morse-Smale"

# Colors
CP_COLORS = {0: '#2166ac', 1: '#4daf4a', 2: '#e41a1c'}
CP_MARKERS = {0: 'o', 1: 's', 2: '^'}


def load_data():
    """Load all required data."""
    xi = pd.read_csv(os.path.join(BASE_PATH, "hypercot_xi.csv"), index_col=0).values
    clean_cp = pd.read_csv(os.path.join(BASE_PATH, "clean_critical_points.csv"))
    noisy_cp = pd.read_csv(os.path.join(BASE_PATH, "noisy_critical_points.csv"))
    clean_vc = pd.read_csv(os.path.join(BASE_PATH, "clean_virtual_centers.csv"))
    noisy_vc = pd.read_csv(os.path.join(BASE_PATH, "noisy_virtual_centers.csv"))
    clean_hyper = pd.read_csv(os.path.join(BASE_PATH, "hypergraph_clean.csv"))
    noisy_hyper = pd.read_csv(os.path.join(BASE_PATH, "noisy_hypergraph.csv")) if os.path.exists(os.path.join(BASE_PATH, "noisy_hypergraph.csv")) else pd.read_csv(os.path.join(BASE_PATH, "hypergraph_noisy.csv"))
    return xi, clean_cp, noisy_cp, clean_vc, noisy_vc, clean_hyper, noisy_hyper


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


def get_region_colors(n_regions):
    """Generate consistent colors for regions."""
    np.random.seed(42)
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    return {i: cmap[i % 20] for i in range(n_regions)}


def draw_ms_complexes(ax, clean_cp, noisy_cp, clean_vc, noisy_vc,
                      clean_hyper, noisy_hyper, clean_colors, noisy_colors=None,
                      noisy_alphas=None, shift_x=120, draw_cps=True, draw_vcs=True):
    """Draw both MS complexes on the same axis."""
    n_clean = len(clean_vc)
    n_noisy = len(noisy_vc)

    # Draw Clean Regions
    for i, row in clean_hyper.iterrows():
        vertices = get_region_polygon(clean_cp, row)
        color = clean_colors[i][:3]
        polygon = Polygon(vertices, closed=True,
                         facecolor=(*color, 0.6), edgecolor='gray', linewidth=0.5)
        ax.add_patch(polygon)

    # Draw Noisy Regions
    for i, row in noisy_hyper.iterrows():
        vertices = get_region_polygon(noisy_cp, row)
        vertices[:, 0] += shift_x

        if noisy_colors is not None:
            color = noisy_colors[i][:3] if isinstance(noisy_colors[i], (list, tuple, np.ndarray)) else noisy_colors[i]
            alpha = noisy_alphas[i] if noisy_alphas is not None else 0.6
            facecolor = (*color, alpha) if isinstance(color, (list, tuple)) else color
        else:
            facecolor = (0.8, 0.8, 0.8, 0.5)

        polygon = Polygon(vertices, closed=True,
                         facecolor=facecolor, edgecolor='gray', linewidth=0.5)
        ax.add_patch(polygon)

    # Draw VCs
    if draw_vcs:
        ax.scatter(clean_vc['vc_x'], clean_vc['vc_y'],
                  c='white', marker='*', s=40, zorder=6,
                  edgecolors='black', linewidths=0.5)
        ax.scatter(noisy_vc['vc_x'] + shift_x, noisy_vc['vc_y'],
                  c='white', marker='*', s=40, zorder=6,
                  edgecolors='black', linewidths=0.5)

    # Draw CPs
    if draw_cps:
        for cp_type in [0, 1, 2]:
            mask = clean_cp['CellDimension'] == cp_type
            ax.scatter(clean_cp.loc[mask, 'Points_0'], clean_cp.loc[mask, 'Points_1'],
                      c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                      s=20, zorder=5, edgecolors='black', linewidths=0.2, alpha=0.8)
            mask = noisy_cp['CellDimension'] == cp_type
            ax.scatter(noisy_cp.loc[mask, 'Points_0'] + shift_x, noisy_cp.loc[mask, 'Points_1'],
                      c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                      s=20, zorder=5, edgecolors='black', linewidths=0.2, alpha=0.8)

    # Dividing line
    ax.axvline(x=shift_x - 10, color='black', linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xlim(-5, shift_x + 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')
    ax.axis('off')


# ============================================================================
# Option A: Row-wise Normalization
# ============================================================================
def visualize_option_A(ax, xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper):
    """Row-wise Normalization: Normalize each row to show relative best match."""
    n_clean = len(clean_vc)
    n_noisy = len(noisy_vc)
    clean_colors = get_region_colors(n_clean)
    shift_x = 120

    # Row-wise normalization
    xi_norm = xi / xi.sum(axis=1, keepdims=True)

    # For noisy regions: use best match with normalized probability as alpha
    noisy_best = np.argmax(xi, axis=0)
    noisy_colors = {}
    noisy_alphas = {}

    for j in range(n_noisy):
        best_i = noisy_best[j]
        # Get row-normalized strength (how much this noisy region matches to best clean)
        row_norm_strength = xi_norm[best_i, j]
        noisy_colors[j] = clean_colors[best_i][:3]
        noisy_alphas[j] = 0.3 + 0.6 * min(row_norm_strength * 5, 1.0)  # Scale up for visibility

    draw_ms_complexes(ax, clean_cp, noisy_cp, clean_vc, noisy_vc,
                      clean_hyper, noisy_hyper, clean_colors, noisy_colors, noisy_alphas,
                      shift_x=shift_x, draw_cps=False)

    # Draw top correspondences based on row-normalized values
    top_k = 10
    flat_norm = xi_norm.flatten()
    top_indices = np.argsort(flat_norm)[-top_k:][::-1]

    for idx in top_indices:
        i, j = np.unravel_index(idx, xi_norm.shape)
        val = xi_norm[i, j]
        alpha = 0.4 + 0.5 * val
        lw = 1 + 3 * val
        ax.plot([clean_vc.iloc[i]['vc_x'], noisy_vc.iloc[j]['vc_x'] + shift_x],
               [clean_vc.iloc[i]['vc_y'], noisy_vc.iloc[j]['vc_y']],
               color=clean_colors[i][:3], alpha=alpha, linewidth=lw, zorder=3)

    ax.set_title('A: Row-wise Normalization\n(Shows relative preference per clean region)',
                 fontsize=9, fontweight='bold')


# ============================================================================
# Option B: Sankey/Alluvial Diagram
# ============================================================================
def visualize_option_B(ax, xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper):
    """Sankey-style flow diagram showing correspondence strength."""
    n_clean = len(clean_vc)
    n_noisy = len(noisy_vc)
    clean_colors = get_region_colors(n_clean)

    # Vertical positions for clean and noisy regions
    clean_y = np.linspace(95, 5, n_clean)
    noisy_y = np.linspace(95, 5, n_noisy)

    clean_x = 20
    noisy_x = 80

    # Draw region bars
    bar_height = 80 / max(n_clean, n_noisy) * 0.6
    for i in range(n_clean):
        rect = Rectangle((clean_x - 3, clean_y[i] - bar_height/2), 6, bar_height,
                         facecolor=clean_colors[i][:3], edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.add_patch(rect)

    for j in range(n_noisy):
        rect = Rectangle((noisy_x - 3, noisy_y[j] - bar_height/2), 6, bar_height,
                         facecolor=(0.7, 0.7, 0.7), edgecolor='black', linewidth=0.5, alpha=0.6)
        ax.add_patch(rect)

    # Draw flows for strongest connections
    max_xi = xi.max()
    threshold = max_xi * 0.3  # Only show strong connections

    for i in range(n_clean):
        for j in range(n_noisy):
            if xi[i, j] > threshold:
                width = 0.5 + 4 * (xi[i, j] / max_xi)
                alpha = 0.3 + 0.5 * (xi[i, j] / max_xi)
                ax.plot([clean_x + 3, noisy_x - 3], [clean_y[i], noisy_y[j]],
                       color=clean_colors[i][:3], alpha=alpha, linewidth=width, zorder=2)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.text(clean_x, 100, f'Clean ({n_clean})', ha='center', fontsize=8, fontweight='bold')
    ax.text(noisy_x, 100, f'Noisy ({n_noisy})', ha='center', fontsize=8, fontweight='bold')
    ax.axis('off')
    ax.set_title('B: Sankey/Alluvial Diagram\n(Flow width = coupling strength)',
                 fontsize=9, fontweight='bold')


# ============================================================================
# Option C: Top-K Lines Only
# ============================================================================
def visualize_option_C(ax, xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper):
    """Show only the top-K strongest correspondence lines."""
    n_clean = len(clean_vc)
    n_noisy = len(noisy_vc)
    clean_colors = get_region_colors(n_clean)
    shift_x = 120

    # Use gray for all noisy regions
    noisy_colors = {j: (0.8, 0.8, 0.8) for j in range(n_noisy)}
    noisy_alphas = {j: 0.4 for j in range(n_noisy)}

    draw_ms_complexes(ax, clean_cp, noisy_cp, clean_vc, noisy_vc,
                      clean_hyper, noisy_hyper, clean_colors, noisy_colors, noisy_alphas,
                      shift_x=shift_x, draw_cps=False)

    # Draw only top-K lines with clear distinction
    top_k = 15
    flat_xi = xi.flatten()
    top_indices = np.argsort(flat_xi)[-top_k:][::-1]
    max_xi = xi.max()

    for rank, idx in enumerate(top_indices):
        i, j = np.unravel_index(idx, xi.shape)
        val = xi[i, j]

        # Stronger distinction: top matches are much more prominent
        alpha = 0.9 - 0.4 * (rank / top_k)
        lw = 4 - 2.5 * (rank / top_k)

        ax.plot([clean_vc.iloc[i]['vc_x'], noisy_vc.iloc[j]['vc_x'] + shift_x],
               [clean_vc.iloc[i]['vc_y'], noisy_vc.iloc[j]['vc_y']],
               color=clean_colors[i][:3], alpha=alpha, linewidth=lw, zorder=3)

        # Annotate top 3
        if rank < 3:
            mid_x = (clean_vc.iloc[i]['vc_x'] + noisy_vc.iloc[j]['vc_x'] + shift_x) / 2
            mid_y = (clean_vc.iloc[i]['vc_y'] + noisy_vc.iloc[j]['vc_y']) / 2
            ax.annotate(f'#{rank+1}', (mid_x, mid_y), fontsize=7, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_title(f'C: Top-{top_k} Lines Only\n(Ranked by coupling strength)',
                 fontsize=9, fontweight='bold')


# ============================================================================
# Option D: Hungarian Assignment + Confidence
# ============================================================================
def visualize_option_D(ax, xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper):
    """One-to-one optimal matching using Hungarian algorithm."""
    n_clean = len(clean_vc)
    n_noisy = len(noisy_vc)
    clean_colors = get_region_colors(n_clean)
    shift_x = 120

    # Hungarian assignment (maximize coupling, so minimize negative)
    cost_matrix = -xi
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Color noisy regions by their assigned clean region
    noisy_colors = {}
    noisy_alphas = {}
    assignments = {}

    for i, j in zip(row_ind, col_ind):
        noisy_colors[j] = clean_colors[i][:3]
        confidence = xi[i, j] / xi.max()
        noisy_alphas[j] = 0.3 + 0.5 * confidence
        assignments[j] = (i, xi[i, j])

    # Unassigned noisy regions (if n_noisy > n_clean)
    for j in range(n_noisy):
        if j not in noisy_colors:
            noisy_colors[j] = (0.5, 0.5, 0.5)
            noisy_alphas[j] = 0.3

    draw_ms_complexes(ax, clean_cp, noisy_cp, clean_vc, noisy_vc,
                      clean_hyper, noisy_hyper, clean_colors, noisy_colors, noisy_alphas,
                      shift_x=shift_x, draw_cps=False)

    # Draw assignment lines
    max_xi = xi.max()
    for i, j in zip(row_ind, col_ind):
        confidence = xi[i, j] / max_xi
        alpha = 0.5 + 0.4 * confidence
        lw = 1 + 2.5 * confidence

        # Use dashed line for low confidence
        linestyle = '-' if confidence > 0.3 else '--'

        ax.plot([clean_vc.iloc[i]['vc_x'], noisy_vc.iloc[j]['vc_x'] + shift_x],
               [clean_vc.iloc[i]['vc_y'], noisy_vc.iloc[j]['vc_y']],
               color=clean_colors[i][:3], alpha=alpha, linewidth=lw,
               linestyle=linestyle, zorder=3)

    ax.set_title('D: Hungarian Assignment\n(Optimal 1-to-1 matching, dashed=low confidence)',
                 fontsize=9, fontweight='bold')


# ============================================================================
# Option E: Bipartite Graph Layout
# ============================================================================
def visualize_option_E(ax, xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper):
    """Bipartite graph with regions as nodes on parallel axes."""
    n_clean = len(clean_vc)
    n_noisy = len(noisy_vc)
    clean_colors = get_region_colors(n_clean)

    # Position nodes on two vertical lines
    clean_x = 20
    noisy_x = 80
    clean_y = np.linspace(90, 10, n_clean)
    noisy_y = np.linspace(90, 10, n_noisy)

    # Draw edges first (connections)
    max_xi = xi.max()
    threshold = max_xi * 0.2

    for i in range(n_clean):
        for j in range(n_noisy):
            if xi[i, j] > threshold:
                strength = xi[i, j] / max_xi
                alpha = 0.2 + 0.6 * strength
                lw = 0.5 + 3 * strength
                ax.plot([clean_x, noisy_x], [clean_y[i], noisy_y[j]],
                       color=clean_colors[i][:3], alpha=alpha, linewidth=lw, zorder=1)

    # Draw nodes
    for i in range(n_clean):
        ax.scatter(clean_x, clean_y[i], c=[clean_colors[i][:3]], s=150,
                  edgecolors='black', linewidths=1, zorder=3)
        ax.text(clean_x - 8, clean_y[i], str(i+1), ha='right', va='center', fontsize=7)

    noisy_best = np.argmax(xi, axis=0)
    for j in range(n_noisy):
        # Color by best match
        color = clean_colors[noisy_best[j]][:3]
        strength = xi[noisy_best[j], j] / max_xi
        ax.scatter(noisy_x, noisy_y[j], c=[color], s=100, alpha=0.3 + 0.6*strength,
                  edgecolors='black', linewidths=0.5, zorder=3)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.text(clean_x, 97, f'Clean\n({n_clean})', ha='center', fontsize=8, fontweight='bold')
    ax.text(noisy_x, 97, f'Noisy\n({n_noisy})', ha='center', fontsize=8, fontweight='bold')
    ax.axis('off')
    ax.set_title('E: Bipartite Graph\n(Node color = best match)', fontsize=9, fontweight='bold')


# ============================================================================
# Option F: Threshold-based with Confidence Intervals
# ============================================================================
def visualize_option_F(ax, xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper):
    """Show only correspondences above threshold with confidence bands."""
    n_clean = len(clean_vc)
    n_noisy = len(noisy_vc)
    clean_colors = get_region_colors(n_clean)
    shift_x = 120

    max_xi = xi.max()
    mean_xi = xi.mean()

    # Classify correspondences
    high_thresh = max_xi * 0.6
    med_thresh = max_xi * 0.3

    noisy_colors = {}
    noisy_alphas = {}

    noisy_best = np.argmax(xi, axis=0)
    noisy_max = np.max(xi, axis=0)

    for j in range(n_noisy):
        best_i = noisy_best[j]
        strength = noisy_max[j]

        if strength > high_thresh:
            noisy_colors[j] = clean_colors[best_i][:3]
            noisy_alphas[j] = 0.8
        elif strength > med_thresh:
            noisy_colors[j] = clean_colors[best_i][:3]
            noisy_alphas[j] = 0.5
        else:
            noisy_colors[j] = (0.7, 0.7, 0.7)
            noisy_alphas[j] = 0.3

    draw_ms_complexes(ax, clean_cp, noisy_cp, clean_vc, noisy_vc,
                      clean_hyper, noisy_hyper, clean_colors, noisy_colors, noisy_alphas,
                      shift_x=shift_x, draw_cps=False)

    # Draw only high-confidence lines (solid) and medium (dashed)
    for i in range(n_clean):
        for j in range(n_noisy):
            if xi[i, j] > high_thresh:
                ax.plot([clean_vc.iloc[i]['vc_x'], noisy_vc.iloc[j]['vc_x'] + shift_x],
                       [clean_vc.iloc[i]['vc_y'], noisy_vc.iloc[j]['vc_y']],
                       color=clean_colors[i][:3], alpha=0.9, linewidth=3,
                       linestyle='-', zorder=4)
            elif xi[i, j] > med_thresh:
                ax.plot([clean_vc.iloc[i]['vc_x'], noisy_vc.iloc[j]['vc_x'] + shift_x],
                       [clean_vc.iloc[i]['vc_y'], noisy_vc.iloc[j]['vc_y']],
                       color=clean_colors[i][:3], alpha=0.5, linewidth=1.5,
                       linestyle='--', zorder=3)

    # Legend
    ax.plot([], [], 'k-', linewidth=3, label='High confidence')
    ax.plot([], [], 'k--', linewidth=1.5, label='Medium confidence')
    ax.legend(loc='lower center', fontsize=7, ncol=2, framealpha=0.9)

    ax.set_title('F: Threshold-based\n(Solid=high, Dashed=medium confidence)',
                 fontsize=9, fontweight='bold')


# ============================================================================
# Option G: Clustered by Similarity Pattern
# ============================================================================
def visualize_option_G(ax, xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper):
    """Group regions by their matching patterns."""
    n_clean = len(clean_vc)
    n_noisy = len(noisy_vc)
    clean_colors = get_region_colors(n_clean)
    shift_x = 120

    # For each clean region, find its dominant matches
    noisy_best = np.argmax(xi, axis=0)

    # Color noisy regions by best match, with saturation based on "exclusivity"
    noisy_colors = {}
    noisy_alphas = {}

    # Count how many noisy regions map to each clean region
    clean_counts = np.bincount(noisy_best, minlength=n_clean)

    for j in range(n_noisy):
        best_i = noisy_best[j]
        noisy_colors[j] = clean_colors[best_i][:3]

        # Higher alpha if this is a "unique" match (clean region has few matches)
        exclusivity = 1.0 / max(clean_counts[best_i], 1)
        # Also consider the coupling strength
        strength = xi[best_i, j] / xi.max()
        noisy_alphas[j] = 0.3 + 0.5 * (exclusivity * 0.5 + strength * 0.5)

    draw_ms_complexes(ax, clean_cp, noisy_cp, clean_vc, noisy_vc,
                      clean_hyper, noisy_hyper, clean_colors, noisy_colors, noisy_alphas,
                      shift_x=shift_x, draw_cps=False)

    # Draw lines only for best matches (one line per noisy region to its best clean)
    for j in range(n_noisy):
        best_i = noisy_best[j]
        strength = xi[best_i, j] / xi.max()

        if strength > 0.2:  # Only show reasonable matches
            alpha = 0.3 + 0.5 * strength
            lw = 0.5 + 2.5 * strength
            ax.plot([clean_vc.iloc[best_i]['vc_x'], noisy_vc.iloc[j]['vc_x'] + shift_x],
                   [clean_vc.iloc[best_i]['vc_y'], noisy_vc.iloc[j]['vc_y']],
                   color=clean_colors[best_i][:3], alpha=alpha, linewidth=lw, zorder=3)

    ax.set_title('G: Best-Match Grouping\n(Color = best matching clean region)',
                 fontsize=9, fontweight='bold')


def main():
    """Create comparison figure with all visualization approaches."""
    print("Loading data...")
    xi, clean_cp, noisy_cp, clean_vc, noisy_vc, clean_hyper, noisy_hyper = load_data()

    print(f"Xi matrix shape: {xi.shape}")
    print(f"Max coupling: {xi.max():.6f}")
    print(f"Mean coupling: {xi.mean():.6f}")

    # Create figure with 7 subplots (2 rows x 4 cols, last spot for legend/info)
    fig = plt.figure(figsize=(20, 10))

    # Grid: 2 rows x 4 columns
    axes = []
    for i in range(7):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(2, 4, i + 1)
        axes.append(ax)

    print("\nGenerating visualizations...")

    # Option A
    print("  A: Row-wise Normalization...")
    visualize_option_A(axes[0], xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper)

    # Option B
    print("  B: Sankey/Alluvial Diagram...")
    visualize_option_B(axes[1], xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper)

    # Option C
    print("  C: Top-K Lines Only...")
    visualize_option_C(axes[2], xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper)

    # Option D
    print("  D: Hungarian Assignment...")
    visualize_option_D(axes[3], xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper)

    # Option E
    print("  E: Bipartite Graph...")
    visualize_option_E(axes[4], xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper)

    # Option F
    print("  F: Threshold-based...")
    visualize_option_F(axes[5], xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper)

    # Option G
    print("  G: Best-Match Grouping...")
    visualize_option_G(axes[6], xi, clean_cp, noisy_cp, clean_vc, noisy_vc,
                       clean_hyper, noisy_hyper)

    # Use last subplot for summary info
    ax_info = fig.add_subplot(2, 4, 8)
    ax_info.axis('off')

    info_text = """
    VISUALIZATION COMPARISON

    Data: {n_clean} clean → {n_noisy} noisy regions
    Max ξ: {max_xi:.4f}
    Mean ξ: {mean_xi:.6f}

    RECOMMENDATIONS:

    • For publication: C or D
      (Clear, focused on strongest matches)

    • For exploration: E or G
      (Shows full correspondence pattern)

    • For flow analysis: B
      (Intuitive, shows many-to-many)

    • For statistical: A or F
      (Normalized/thresholded values)
    """.format(
        n_clean=len(clean_vc),
        n_noisy=len(noisy_vc),
        max_xi=xi.max(),
        mean_xi=xi.mean()
    )

    ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Hypergraph Correspondence Visualization Comparison',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(BASE_PATH, "hypergraph_visualization_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {save_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
