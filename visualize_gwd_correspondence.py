"""
GWD Correspondence Visualization
Matching the style of wd_correspondence_refined.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from scipy.sparse.csgraph import shortest_path
import ot
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH = "/Users/gmeng/Desktop/COOT on Morse-Smale"

# Visual settings - matching WD style
CP_COLORS = {
    0: '#2166ac',  # min - blue
    1: '#4daf4a',  # saddle - green
    2: '#e41a1c'   # max - red
}

CP_MARKERS = {
    0: 'o',  # min - circle
    1: 's',  # saddle - square
    2: '^'   # max - triangle
}

CP_LABELS = {0: 'Minimum', 1: 'Saddle', 2: 'Maximum'}

# Block background colors (same type = green, different = pink)
BLOCK_SAME = '#d4edda'    # light green
BLOCK_DIFF = '#f8d7da'    # light pink


# =============================================================================
# DATA LOADING
# =============================================================================

def load_critical_points(filepath):
    """Load critical points from CSV."""
    df = pd.read_csv(filepath)
    result = pd.DataFrame({
        'point_id': df['Point ID'],
        'cell_id': df['CellId'],
        'cp_type': df['CellDimension'],
        'x': df['Points_0'],
        'y': df['Points_1'],
        'z': df['Points_2'],
        'scalar': df['ttkVertexScalarField']
    })
    return result


def build_adjacency_from_separatrices(cp_df, sep_cells_df):
    """Build adjacency matrix from separatrices_cells.csv."""
    n_cp = len(cp_df)
    adjacency = np.zeros((n_cp, n_cp))
    cell_id_to_idx = {cid: idx for idx, cid in enumerate(cp_df['cell_id'].values)}

    unique_sep = sep_cells_df.drop_duplicates(subset=['SeparatrixId'])[
        ['SeparatrixId', 'SourceId', 'DestinationId']
    ]

    for _, row in unique_sep.iterrows():
        src = int(row['SourceId'])
        dst = int(row['DestinationId'])
        if src in cell_id_to_idx and dst in cell_id_to_idx:
            i = cell_id_to_idx[src]
            j = cell_id_to_idx[dst]
            adjacency[i, j] = 1
            adjacency[j, i] = 1

    return adjacency


def compute_gromov_wasserstein(cp1, cp2, adj1, adj2):
    """Compute GWD between two MS complexes using graph geodesic with Euclidean edge weights."""
    n1, n2 = len(cp1), len(cp2)

    # Get spatial coordinates
    coords1 = np.column_stack([cp1['x'].values, cp1['y'].values, cp1['z'].values])
    coords2 = np.column_stack([cp2['x'].values, cp2['y'].values, cp2['z'].values])

    # Build weighted adjacency matrices (edge weight = Euclidean distance)
    from scipy.spatial.distance import cdist
    euclidean1 = cdist(coords1, coords1, metric='euclidean')
    euclidean2 = cdist(coords2, coords2, metric='euclidean')

    # Create weighted graph: edge weight = Euclidean distance if connected, inf otherwise
    weighted_adj1 = np.where(adj1 > 0, euclidean1, np.inf)
    weighted_adj2 = np.where(adj2 > 0, euclidean2, np.inf)
    np.fill_diagonal(weighted_adj1, 0)
    np.fill_diagonal(weighted_adj2, 0)

    # Compute shortest path distances using Dijkstra
    D1 = shortest_path(weighted_adj1, directed=False, method='D')
    D2 = shortest_path(weighted_adj2, directed=False, method='D')

    # Handle disconnected components (inf distances)
    max_dist1 = D1[np.isfinite(D1)].max() if np.any(np.isfinite(D1)) else 1
    max_dist2 = D2[np.isfinite(D2)].max() if np.any(np.isfinite(D2)) else 1
    D1[np.isinf(D1)] = max_dist1 * 2  # Set disconnected pairs to 2x max distance
    D2[np.isinf(D2)] = max_dist2 * 2

    # Normalize to [0, 1]
    D1 = D1 / (D1.max() + 1e-8)
    D2 = D2 / (D2.max() + 1e-8)

    p = np.ones(n1) / n1
    q = np.ones(n2) / n2

    coupling, log = ot.gromov.gromov_wasserstein(
        D1, D2, p, q, loss_fun='square_loss', log=True
    )

    return log['gw_dist'], coupling


def compute_edge_preservation(adj1, adj2, coupling):
    """Compute edge preservation: check if edges in clean map to edges in noisy."""
    edges1 = get_edges(adj1)
    edges2 = get_edges(adj2)

    n2 = adj2.shape[0]

    preserved_edges = []
    broken_edges = []

    for (u, v) in edges1:
        # Find best match for u and v in noisy
        u_prime = np.argmax(coupling[u, :])
        v_prime = np.argmax(coupling[v, :])

        # Check if (u', v') is an edge in noisy
        if adj2[u_prime, v_prime] > 0:
            preserved_edges.append((u, v, u_prime, v_prime))
        else:
            broken_edges.append((u, v, u_prime, v_prime))

    return preserved_edges, broken_edges, edges1, edges2


def load_data():
    """Load all data and compute GWD."""
    print("Loading data...")

    cp1 = load_critical_points(os.path.join(BASE_PATH, "clean_critical_points.csv"))
    cp2 = load_critical_points(os.path.join(BASE_PATH, "noisy_critical_points.csv"))

    print(f"  Clean: {len(cp1)} CPs (min={sum(cp1['cp_type']==0)}, sad={sum(cp1['cp_type']==1)}, max={sum(cp1['cp_type']==2)})")
    print(f"  Noisy: {len(cp2)} CPs (min={sum(cp2['cp_type']==0)}, sad={sum(cp2['cp_type']==1)}, max={sum(cp2['cp_type']==2)})")

    sep1 = pd.read_csv(os.path.join(BASE_PATH, "clean_separatrices_cells.csv"))
    sep2 = pd.read_csv(os.path.join(BASE_PATH, "noisy_separatrices_cells.csv"))

    print("Building adjacency matrices...")
    adj1 = build_adjacency_from_separatrices(cp1, sep1)
    adj2 = build_adjacency_from_separatrices(cp2, sep2)

    n_edges1 = int(np.sum(adj1) / 2)
    n_edges2 = int(np.sum(adj2) / 2)
    print(f"  Clean edges: {n_edges1}")
    print(f"  Noisy edges: {n_edges2}")

    print("Computing Gromov-Wasserstein distance...")
    gwd_dist, gwd_coupling = compute_gromov_wasserstein(cp1, cp2, adj1, adj2)
    print(f"  GWD = {gwd_dist:.4f}")

    # Compute edge preservation
    print("Computing edge preservation...")
    preserved, broken, edges1, edges2 = compute_edge_preservation(adj1, adj2, gwd_coupling)
    edge_pres_rate = len(preserved) / len(edges1) * 100 if edges1 else 0
    print(f"  Edge preservation: {len(preserved)}/{len(edges1)} ({edge_pres_rate:.1f}%)")

    return {
        'cp1': cp1, 'cp2': cp2,
        'adj1': adj1, 'adj2': adj2,
        'gwd_dist': gwd_dist,
        'gwd_coupling': gwd_coupling,
        'preserved_edges': preserved,
        'broken_edges': broken,
        'edges1': edges1,
        'edges2': edges2
    }


def get_edges(adj):
    """Extract edge list from adjacency matrix."""
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                edges.append((i, j))
    return edges


# =============================================================================
# VISUALIZATION (matching wd_correspondence_refined.png style)
# =============================================================================

def plot_gwd_correspondence(data, save_path, n_top=40):
    """Create GWD correspondence visualization matching WD refined style."""
    print("\nCreating GWD correspondence visualization...")

    cp1, cp2 = data['cp1'], data['cp2']
    adj1, adj2 = data['adj1'], data['adj2']
    coupling = data['gwd_coupling']
    gwd_dist = data['gwd_dist']
    preserved_edges = data['preserved_edges']
    broken_edges = data['broken_edges']

    types1 = cp1['cp_type'].values
    types2 = cp2['cp_type'].values
    n1, n2 = len(cp1), len(cp2)

    # Edge stats
    n_edges1 = len(preserved_edges) + len(broken_edges)
    n_preserved = len(preserved_edges)
    edge_pres_rate = n_preserved / n_edges1 * 100 if n_edges1 > 0 else 0

    # Sort by type for visualization
    sort_idx1 = np.argsort(types1)
    sort_idx2 = np.argsort(types2)

    coupling_sorted = coupling[sort_idx1, :][:, sort_idx2]
    types1_sorted = types1[sort_idx1]
    types2_sorted = types2[sort_idx2]

    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.8, 0.8], wspace=0.12,
                          left=0.05, right=0.95, top=0.85, bottom=0.18)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ==========================================================================
    # LEFT PANEL: Coupling Matrix with type block backgrounds
    # ==========================================================================

    # Get type boundaries
    def get_boundaries(types_sorted):
        bounds = {}
        for t in [0, 1, 2]:
            idx = np.where(types_sorted == t)[0]
            if len(idx) > 0:
                bounds[t] = (idx[0], idx[-1] + 1)
        return bounds

    bounds1 = get_boundaries(types1_sorted)
    bounds2 = get_boundaries(types2_sorted)

    # Draw block backgrounds
    for t1, (y0, y1) in bounds1.items():
        for t2, (x0, x1) in bounds2.items():
            color = BLOCK_SAME if t1 == t2 else BLOCK_DIFF
            rect = plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                  facecolor=color, edgecolor='none', zorder=0)
            ax1.add_patch(rect)

    # Plot coupling matrix with log scale
    coupling_plot = coupling_sorted.copy()
    coupling_plot[coupling_plot == 0] = np.nan  # Hide zeros

    vmin = np.nanmin(coupling_plot[coupling_plot > 0]) if np.any(coupling_plot > 0) else 1e-4
    vmax = np.nanmax(coupling_plot)

    im = ax1.imshow(coupling_plot, cmap='YlOrRd', aspect='auto',
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    interpolation='nearest', zorder=1)

    # Draw divider lines
    for t, (start, end) in bounds1.items():
        if start > 0:
            ax1.axhline(y=start - 0.5, color='black', linewidth=1.0)
    for t, (start, end) in bounds2.items():
        if start > 0:
            ax1.axvline(x=start - 0.5, color='black', linewidth=1.0)

    # Type labels on y-axis (left) - moved further left to avoid overlap
    type_labels = ['Min', 'Sad', 'Max']
    for t in [0, 1, 2]:
        if t in bounds1:
            mid = (bounds1[t][0] + bounds1[t][1]) / 2
            ax1.text(-5, mid, type_labels[t], ha='right', va='center',
                    fontsize=11, fontweight='bold', color=CP_COLORS[t])

    # Type labels on x-axis (bottom)
    for t in [0, 1, 2]:
        if t in bounds2:
            mid = (bounds2[t][0] + bounds2[t][1]) / 2
            ax1.text(mid, n1 + 3, type_labels[t], ha='center', va='top',
                    fontsize=11, fontweight='bold', color=CP_COLORS[t])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Transport Mass (log scale)', fontsize=10)

    # Axis settings
    ax1.set_xlim(-0.5, n2 - 0.5)
    ax1.set_ylim(n1 - 0.5, -0.5)
    ax1.set_xlabel('Noisy CP Index', fontsize=11, labelpad=25)
    ax1.set_ylabel('Clean CP Index', fontsize=11)
    ax1.set_title('GWD Coupling Matrix', fontsize=13, fontweight='bold')

    # ==========================================================================
    # RIGHT PANEL: CP Correspondences (vertical layout by type)
    # ==========================================================================

    # Get top correspondences
    flat_idx = np.argsort(coupling.ravel())[::-1][:n_top]
    top_pairs = [(idx // n2, idx % n2, coupling.ravel()[idx]) for idx in flat_idx]

    # Count type matches/mismatches
    n_match = sum(1 for i, j, _ in top_pairs if types1[i] == types2[j])
    n_mismatch = n_top - n_match
    type_pres = n_match / n_top * 100

    # Sort CPs by type for vertical layout (Max=top, Min=bottom)
    # Create y-positions: Max at top, then Sad, then Min
    def get_y_positions(types):
        n = len(types)
        y_pos = np.zeros(n)

        # Count each type
        n_min = sum(types == 0)
        n_sad = sum(types == 1)
        n_max = sum(types == 2)

        # Assign y positions (Max at top)
        y_max_start = 0
        y_sad_start = n_max
        y_min_start = n_max + n_sad

        max_idx, sad_idx, min_idx = 0, 0, 0
        for i, t in enumerate(types):
            if t == 2:  # Max
                y_pos[i] = y_max_start + max_idx
                max_idx += 1
            elif t == 1:  # Sad
                y_pos[i] = y_sad_start + sad_idx
                sad_idx += 1
            else:  # Min
                y_pos[i] = y_min_start + min_idx
                min_idx += 1

        return y_pos, n_max, n_sad, n_min

    y1, n_max1, n_sad1, n_min1 = get_y_positions(types1)
    y2, n_max2, n_sad2, n_min2 = get_y_positions(types2)

    x1_pos = 0.2
    x2_pos = 0.8

    # Draw correspondence lines
    for i, j, weight in top_pairs:
        if types1[i] == types2[j]:
            color = '#2ca02c'  # green
        else:
            color = '#d62728'  # red
        ax2.plot([x1_pos, x2_pos], [y1[i], y2[j]],
                color=color, linewidth=1.0, alpha=0.7, zorder=1)

    # Draw type divider lines
    ax2.axhline(y=n_max1 - 0.5, color='gray', linestyle='--', alpha=0.5, xmin=0.05, xmax=0.35)
    ax2.axhline(y=n_max1 + n_sad1 - 0.5, color='gray', linestyle='--', alpha=0.5, xmin=0.05, xmax=0.35)
    ax2.axhline(y=n_max2 - 0.5, color='gray', linestyle='--', alpha=0.5, xmin=0.65, xmax=0.95)
    ax2.axhline(y=n_max2 + n_sad2 - 0.5, color='gray', linestyle='--', alpha=0.5, xmin=0.65, xmax=0.95)

    # Draw CPs
    for i in range(n1):
        ax2.scatter(x1_pos, y1[i], c=CP_COLORS[types1[i]],
                   s=50, marker=CP_MARKERS[types1[i]], zorder=3,
                   edgecolors='white', linewidths=0.5)

    for j in range(n2):
        ax2.scatter(x2_pos, y2[j], c=CP_COLORS[types2[j]],
                   s=50, marker=CP_MARKERS[types2[j]], zorder=3,
                   edgecolors='white', linewidths=0.5)

    # Type labels on left
    ax2.text(0.05, n_max1 / 2, 'Max', ha='center', va='center', fontsize=11,
             fontweight='bold', color=CP_COLORS[2])
    ax2.text(0.05, n_max1 + n_sad1 / 2, 'Sad', ha='center', va='center', fontsize=11,
             fontweight='bold', color=CP_COLORS[1])
    ax2.text(0.05, n_max1 + n_sad1 + n_min1 / 2, 'Min', ha='center', va='center', fontsize=11,
             fontweight='bold', color=CP_COLORS[0])

    # Type labels on right
    ax2.text(0.95, n_max2 / 2, 'Max', ha='center', va='center', fontsize=11,
             fontweight='bold', color=CP_COLORS[2])
    ax2.text(0.95, n_max2 + n_sad2 / 2, 'Sad', ha='center', va='center', fontsize=11,
             fontweight='bold', color=CP_COLORS[1])
    ax2.text(0.95, n_max2 + n_sad2 + n_min2 / 2, 'Min', ha='center', va='center', fontsize=11,
             fontweight='bold', color=CP_COLORS[0])

    # Labels
    ax2.text(x1_pos, n1 + 1.5, f'Clean ({n1})', ha='center', fontsize=11, fontweight='bold')
    ax2.text(x2_pos, n2 + 1.5, f'Noisy ({n2})', ha='center', fontsize=11, fontweight='bold')

    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(max(n1, n2) + 8, -2)  # More room at bottom for legend
    ax2.axis('off')
    ax2.set_title(f'CP Correspondences (Top {n_top})', fontsize=13, fontweight='bold')

    # Legend - place at bottom center, horizontal layout
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CP_COLORS[0],
                   markersize=7, label='Min'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=CP_COLORS[1],
                   markersize=7, label='Sad'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=CP_COLORS[2],
                   markersize=7, label='Max'),
        plt.Line2D([0], [0], color='#2ca02c', linewidth=2, label='Match'),
        plt.Line2D([0], [0], color='#d62728', linewidth=2, label='Mismatch'),
    ]
    ax2.legend(handles=legend_elements, loc='lower center', fontsize=8,
               framealpha=0.95, ncol=5, bbox_to_anchor=(0.5, 0.02))

    # ==========================================================================
    # RIGHT PANEL (ax3): Edge Preservation - Two Graph Views
    # ==========================================================================

    # Use spatial coordinates (x, y) from the data for graph layout
    x1_coords = cp1['x'].values
    y1_coords = cp1['y'].values
    x2_coords = cp2['x'].values
    y2_coords = cp2['y'].values

    # Normalize coordinates to [0, 1] for plotting
    def normalize_coords(x, y):
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        return x_norm, y_norm

    x1_norm, y1_norm = normalize_coords(x1_coords, y1_coords)
    x2_norm, y2_norm = normalize_coords(x2_coords, y2_coords)

    # Create set of preserved edge endpoints for quick lookup
    preserved_set = set((u, v) for (u, v, _, _) in preserved_edges)
    preserved_set.update((v, u) for (u, v, _, _) in preserved_edges)

    # Split ax3 into two subplots (clean graph on top, noisy graph on bottom)
    # Draw Clean Graph (top half) - leave more gap in middle for label
    y_offset_clean = 0.58
    y_scale = 0.35
    x_scale = 0.9
    x_offset = 0.05

    # Draw edges for clean graph
    for (u, v) in data['edges1']:
        if (u, v) in preserved_set:
            color = '#2ca02c'  # green - preserved
            alpha = 0.7
        else:
            color = '#d62728'  # red - broken
            alpha = 0.7
        ax3.plot([x_offset + x1_norm[u] * x_scale, x_offset + x1_norm[v] * x_scale],
                [y_offset_clean + y1_norm[u] * y_scale, y_offset_clean + y1_norm[v] * y_scale],
                color=color, linewidth=1.2, alpha=alpha, zorder=1)

    # Draw CPs for clean graph
    for i in range(n1):
        ax3.scatter(x_offset + x1_norm[i] * x_scale, y_offset_clean + y1_norm[i] * y_scale,
                   c=CP_COLORS[types1[i]], s=50, marker=CP_MARKERS[types1[i]], zorder=3,
                   edgecolors='black', linewidths=0.5)

    # Clean graph label
    ax3.text(0.5, 0.98, f'Clean Graph ({n_edges1} edges)', ha='center', va='top',
             fontsize=11, fontweight='bold', transform=ax3.transAxes)

    # Draw Noisy Graph (bottom half) - adjusted for gap
    y_offset_noisy = 0.08

    # For noisy graph, show all edges in gray (structural reference)
    for (u, v) in data['edges2']:
        ax3.plot([x_offset + x2_norm[u] * x_scale, x_offset + x2_norm[v] * x_scale],
                [y_offset_noisy + y2_norm[u] * y_scale, y_offset_noisy + y2_norm[v] * y_scale],
                color='#888888', linewidth=0.8, alpha=0.5, zorder=1)

    # Draw CPs for noisy graph
    for j in range(n2):
        ax3.scatter(x_offset + x2_norm[j] * x_scale, y_offset_noisy + y2_norm[j] * y_scale,
                   c=CP_COLORS[types2[j]], s=50, marker=CP_MARKERS[types2[j]], zorder=3,
                   edgecolors='black', linewidths=0.5)

    # Noisy graph label - positioned in the gap between graphs
    ax3.text(0.5, 0.50, f'Noisy Graph ({len(data["edges2"])} edges)', ha='center', va='center',
             fontsize=11, fontweight='bold', transform=ax3.transAxes,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

    # Divider line
    ax3.axhline(y=0.50, color='gray', linestyle='-', alpha=0.3, linewidth=1, zorder=0)

    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.02)
    ax3.axis('off')
    ax3.set_title(f'Edge Preservation: {n_preserved}/{n_edges1} ({edge_pres_rate:.1f}%)',
                  fontsize=13, fontweight='bold')

    # Legend for edge preservation - place at bottom, horizontal
    edge_legend = [
        plt.Line2D([0], [0], color='#2ca02c', linewidth=2, label=f'Preserved ({n_preserved})'),
        plt.Line2D([0], [0], color='#d62728', linewidth=2, label=f'Broken ({len(broken_edges)})'),
        plt.Line2D([0], [0], color='#888888', linewidth=2, label='Noisy edges'),
    ]
    ax3.legend(handles=edge_legend, loc='lower center', fontsize=8,
               framealpha=0.95, ncol=3, bbox_to_anchor=(0.5, 0.02))

    # ==========================================================================
    # Footer info
    # ==========================================================================
    footer = (f"GWD = {gwd_dist:.4f}  |  "
              f"Clean: {n1} CPs, {n_edges1} edges  |  "
              f"Noisy: {n2} CPs, {len(data['edges2'])} edges  |  "
              f"Type-preserving: {type_pres:.1f}%  |  "
              f"Edge preservation: {edge_pres_rate:.1f}%")

    fig.text(0.5, 0.06, footer, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#ffffcc', edgecolor='gray', alpha=0.9))

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {save_path}")
    print(f"  Type-preserving: {type_pres:.1f}% ({n_match}/{n_top})")
    print(f"  Edge preservation: {edge_pres_rate:.1f}% ({n_preserved}/{n_edges1})")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    data = load_data()
    plot_gwd_correspondence(
        data,
        os.path.join(BASE_PATH, "gwd_point_edge_correspondence.png"),
        n_top=40
    )
    print("\nDone!")
