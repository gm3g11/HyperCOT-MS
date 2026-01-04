"""
Visualize VC (Virtual Center) Adjacency for Augmented Hypergraph

Generates a 3-panel figure:
1. Left: Virtual Center Generation (intersection of min-max and saddle-saddle lines)
2. Middle: Augmented Graph with CP↔CP, CP↔VC, VC↔VC edges
3. Right: Example Shortest Path from CP to Region's VC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import os

BASE_PATH = "/Users/gmeng/Desktop/COOT on Morse-Smale"

# Colors
CP_COLORS = {0: '#2166ac', 1: '#4daf4a', 2: '#e41a1c'}  # min, saddle, max
CP_LABELS = {0: 'Minimum', 1: 'Saddle', 2: 'Maximum'}
CP_MARKERS = {0: 'o', 1: 's', 2: '^'}


def load_data(prefix):
    """Load all required data for visualization."""
    # Critical points
    cp_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_critical_points.csv"))
    cp_data = pd.DataFrame({
        'point_id': cp_df['Point ID'],
        'cell_id': cp_df['CellId'],
        'cp_type': cp_df['CellDimension'],
        'x': cp_df['Points_0'],
        'y': cp_df['Points_1'],
        'z': cp_df['Points_2'],
    })

    # Separatrices (for CP↔CP edges)
    sep_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_separatrices_cells.csv"))
    cell_id_to_idx = {cid: idx for idx, cid in enumerate(cp_data['cell_id'].values)}
    unique_sep = sep_df.drop_duplicates(subset=['SeparatrixId'])[['SourceId', 'DestinationId']]

    edges = []
    for _, row in unique_sep.iterrows():
        src, dst = int(row['SourceId']), int(row['DestinationId'])
        if src in cell_id_to_idx and dst in cell_id_to_idx:
            edges.append((cell_id_to_idx[src], cell_id_to_idx[dst]))

    # Hypergraph
    hyper_df = pd.read_csv(os.path.join(BASE_PATH, f"hypergraph_{prefix}.csv"))

    # Virtual centers
    vc_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_virtual_centers.csv"))

    # VC adjacency
    vc_adj_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_vc_adjacency.csv"))

    return cp_data, edges, hyper_df, vc_df, vc_adj_df


def compute_virtual_center(cp_data, min_id, max_id, saddle_ids):
    """Compute virtual center as intersection of (min,max) and (saddle,saddle) lines."""
    min_pos = cp_data.iloc[min_id][['x', 'y', 'z']].values.astype(float)
    max_pos = cp_data.iloc[max_id][['x', 'y', 'z']].values.astype(float)
    s1_pos = cp_data.iloc[saddle_ids[0]][['x', 'y', 'z']].values.astype(float)
    s2_pos = cp_data.iloc[saddle_ids[1]][['x', 'y', 'z']].values.astype(float)

    return min_pos, max_pos, s1_pos, s2_pos


def build_augmented_graph(cp_data, edges, hyper_df, vc_df):
    """Build augmented graph and return adjacency info for visualization."""
    n_cp = len(cp_data)
    n_hyper = len(hyper_df)
    n_total = n_cp + n_hyper

    coords = cp_data[['x', 'y', 'z']].values
    vc_coords = vc_df[['vc_x', 'vc_y', 'vc_z']].values

    # Initialize adjacency matrix
    adj = np.full((n_total, n_total), np.inf)
    np.fill_diagonal(adj, 0)

    # Store edges for visualization
    cp_cp_edges = []
    cp_vc_edges = []
    vc_vc_edges = []
    vc_vc_edges_strong = []  # Edges with 2+ shared CPs

    # 1. CP ↔ CP edges (separatrices)
    for i, j in edges:
        dist = np.linalg.norm(coords[i] - coords[j])
        adj[i, j] = dist
        adj[j, i] = dist
        cp_cp_edges.append((i, j))

    # 2. CP ↔ VC edges (boundary membership)
    for h_idx, row in hyper_df.iterrows():
        vc_node = n_cp + h_idx
        vc_pos = vc_coords[h_idx]

        # Get boundary CPs
        hyperedge = eval(row['hyperedge'])
        for cp_idx in hyperedge:
            dist = np.linalg.norm(coords[cp_idx] - vc_pos)
            adj[cp_idx, vc_node] = dist
            adj[vc_node, cp_idx] = dist
            cp_vc_edges.append((cp_idx, h_idx))

    # 3. VC ↔ VC edges (adjacent regions)
    hyperedge_members = [eval(row['hyperedge']) for _, row in hyper_df.iterrows()]
    for i in range(n_hyper):
        members_i = set(hyperedge_members[i])
        for j in range(i + 1, n_hyper):
            members_j = set(hyperedge_members[j])
            shared = members_i & members_j
            if len(shared) > 0:
                vc_i = n_cp + i
                vc_j = n_cp + j
                dist = np.linalg.norm(vc_coords[i] - vc_coords[j])
                adj[vc_i, vc_j] = dist
                adj[vc_j, vc_i] = dist
                vc_vc_edges.append((i, j))
                if len(shared) >= 2:
                    vc_vc_edges_strong.append((i, j))

    return adj, cp_cp_edges, cp_vc_edges, vc_vc_edges, vc_vc_edges_strong, n_cp, n_hyper


def find_shortest_path(adj, source, target, n_cp):
    """Find shortest path and return path nodes."""
    dist_matrix, predecessors = dijkstra(
        csr_matrix(adj), directed=False,
        indices=source, return_predecessors=True
    )

    # Reconstruct path
    path = []
    current = target
    while current != source and current >= 0:
        path.append(current)
        current = predecessors[current]
    path.append(source)
    path = path[::-1]

    return path, dist_matrix[target]


def plot_vc_adjacency(prefix, save_path):
    """Generate the 3-panel VC adjacency visualization."""

    # Load data
    cp_data, edges, hyper_df, vc_df, vc_adj_df = load_data(prefix)

    # Build augmented graph
    adj, cp_cp_edges, cp_vc_edges, vc_vc_edges, vc_vc_edges_strong, n_cp, n_hyper = build_augmented_graph(
        cp_data, edges, hyper_df, vc_df
    )

    coords = cp_data[['x', 'y']].values
    types = cp_data['cp_type'].values
    vc_coords = vc_df[['vc_x', 'vc_y']].values

    # Create figure with more height for legends at bottom
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # =========================================================================
    # Panel 1: Virtual Center Generation
    # =========================================================================
    ax1 = axes[0]

    # Choose a region near the center of the domain for better visualization
    # Find region whose VC is closest to center (50, 50)
    center = np.array([50, 50])
    vc_distances_to_center = np.linalg.norm(vc_coords - center, axis=1)
    example_region = np.argmin(vc_distances_to_center)

    row = hyper_df.iloc[example_region]
    min_id = int(row['min_id'])
    max_id = int(row['max_id'])
    saddle_ids = eval(row['boundary_saddles'])

    min_pos, max_pos, s1_pos, s2_pos = compute_virtual_center(
        cp_data, min_id, max_id, saddle_ids
    )
    vc_pos = vc_coords[example_region]

    # Plot all CPs (faded)
    for cp_type in [0, 1, 2]:
        mask = types == cp_type
        ax1.scatter(coords[mask, 0], coords[mask, 1],
                   c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                   s=30, alpha=0.2, edgecolors='none')

    # Highlight the 4 CPs of the example region
    ax1.scatter(min_pos[0], min_pos[1], c=CP_COLORS[0], marker=CP_MARKERS[0],
               s=150, edgecolors='black', linewidths=2, zorder=10, label='Minimum')
    ax1.scatter(max_pos[0], max_pos[1], c=CP_COLORS[2], marker=CP_MARKERS[2],
               s=150, edgecolors='black', linewidths=2, zorder=10, label='Maximum')
    ax1.scatter(s1_pos[0], s1_pos[1], c=CP_COLORS[1], marker=CP_MARKERS[1],
               s=150, edgecolors='black', linewidths=2, zorder=10, label='Saddle')
    ax1.scatter(s2_pos[0], s2_pos[1], c=CP_COLORS[1], marker=CP_MARKERS[1],
               s=150, edgecolors='black', linewidths=2, zorder=10)

    # Draw min-max line (dashed blue)
    ax1.plot([min_pos[0], max_pos[0]], [min_pos[1], max_pos[1]],
            'b--', linewidth=2, alpha=0.8, label='Min-Max line')

    # Draw saddle-saddle line (dashed green)
    ax1.plot([s1_pos[0], s2_pos[0]], [s1_pos[1], s2_pos[1]],
            'g--', linewidth=2, alpha=0.8, label='Saddle-Saddle line')

    # Plot virtual center (intersection)
    ax1.scatter(vc_pos[0], vc_pos[1], c='orange', marker='*', s=400,
               edgecolors='black', linewidths=2, zorder=15, label='Virtual Center')

    # Add annotation - position text to avoid overlapping with CPs
    ax1.annotate(f'VC (Region {example_region + 1})',
                xy=(vc_pos[0], vc_pos[1]),
                xytext=(vc_pos[0] + 15, vc_pos[1] + 20),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-5, 105)
    ax1.set_xlabel('X', fontsize=11)
    ax1.set_ylabel('Y', fontsize=11)
    ax1.set_title(f'Virtual Center Generation\n(Region {example_region + 1}: Intersection of Min-Max and Saddle-Saddle lines)',
                 fontsize=11, fontweight='bold')
    ax1.legend(loc='lower center', fontsize=8, framealpha=0.95,
               ncol=2, bbox_to_anchor=(0.5, -0.18))
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle=':')

    # =========================================================================
    # Panel 2: Augmented Graph with All Edge Types
    # =========================================================================
    ax2 = axes[1]

    # Draw CP↔CP edges (gray, thin)
    for i, j in cp_cp_edges:
        ax2.plot([coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color='gray', linewidth=0.8, alpha=0.5, zorder=1)

    # Draw VC↔VC edges - only strong ones (2+ shared CPs) for cleaner visualization
    for i, j in vc_vc_edges_strong:
        ax2.plot([vc_coords[i, 0], vc_coords[j, 0]],
                [vc_coords[i, 1], vc_coords[j, 1]],
                color='orange', linewidth=1.5, alpha=0.7, zorder=2)

    # Draw CP↔VC edges (dotted, very light - reduced for clarity)
    for cp_idx, vc_idx in cp_vc_edges:
        ax2.plot([coords[cp_idx, 0], vc_coords[vc_idx, 0]],
                [coords[cp_idx, 1], vc_coords[vc_idx, 1]],
                color='purple', linewidth=0.3, alpha=0.15, linestyle=':', zorder=1)

    # Plot CPs
    for cp_type in [0, 1, 2]:
        mask = types == cp_type
        ax2.scatter(coords[mask, 0], coords[mask, 1],
                   c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                   s=50, edgecolors='black', linewidths=0.5, zorder=5)

    # Plot VCs
    ax2.scatter(vc_coords[:, 0], vc_coords[:, 1],
               c='orange', marker='*', s=100,
               edgecolors='black', linewidths=0.5, zorder=5)

    # Create legend
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=1.5, label='CP↔CP (separatrix)'),
        Line2D([0], [0], color='orange', linewidth=2, label='VC↔VC (adjacent)'),
        Line2D([0], [0], color='purple', linewidth=1, linestyle=':', label='CP↔VC (boundary)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='orange',
               markersize=10, label='Virtual Centers'),
    ]

    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-5, 105)
    ax2.set_xlabel('X', fontsize=11)
    ax2.set_ylabel('Y', fontsize=11)
    ax2.set_title(f'Augmented Graph with VC-VC Adjacency\n(Orange = {len(vc_vc_edges_strong)} strong adj. pairs sharing 2+ CPs)',
                 fontsize=11, fontweight='bold')
    ax2.legend(handles=legend_elements, loc='lower center', fontsize=8,
               framealpha=0.95, ncol=2, bbox_to_anchor=(0.5, -0.15))
    ax2.set_aspect('equal')

    # =========================================================================
    # Panel 3: Example Shortest Path
    # =========================================================================
    ax3 = axes[2]

    # Use specific examples for better visualization
    if prefix == "clean":
        # For clean: CP 45 to Region 20
        source_cp = 45
        target_region = 19  # Region 20 (0-indexed)
    else:
        # For noisy: use example_region from Panel 1, find far CP
        target_region = example_region
        cp_distances_to_vc = np.linalg.norm(coords - vc_coords[target_region], axis=1)
        source_cp = np.argmax(cp_distances_to_vc)

    target_vc_node = n_cp + target_region

    # Find shortest path
    path, path_dist = find_shortest_path(adj, source_cp, target_vc_node, n_cp)

    # Draw all edges (very faded)
    for i, j in cp_cp_edges:
        ax3.plot([coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color='gray', linewidth=0.5, alpha=0.2, zorder=1)

    for i, j in vc_vc_edges:
        ax3.plot([vc_coords[i, 0], vc_coords[j, 0]],
                [vc_coords[i, 1], vc_coords[j, 1]],
                color='orange', linewidth=0.8, alpha=0.3, zorder=1)

    for cp_idx, vc_idx in cp_vc_edges:
        ax3.plot([coords[cp_idx, 0], vc_coords[vc_idx, 0]],
                [coords[cp_idx, 1], vc_coords[vc_idx, 1]],
                color='purple', linewidth=0.3, alpha=0.2, linestyle=':', zorder=1)

    # Draw the shortest path (highlighted)
    path_coords = []
    path_labels = []
    for node in path:
        if node < n_cp:
            path_coords.append(coords[node])
            path_labels.append(f'CP {node}')
        else:
            vc_idx = node - n_cp
            path_coords.append(vc_coords[vc_idx])
            path_labels.append(f'VC{vc_idx + 1}')

    path_coords = np.array(path_coords)

    # Draw path edges with different colors based on edge type
    for i in range(len(path) - 1):
        node1, node2 = path[i], path[i + 1]

        # Determine edge type and color
        if node1 < n_cp and node2 < n_cp:
            color = 'blue'
            label = 'CP↔CP edge'
        elif node1 < n_cp or node2 < n_cp:
            color = 'green'
            label = 'CP↔VC edge'
        else:
            color = 'red'
            label = 'VC↔VC edge'

        ax3.plot([path_coords[i, 0], path_coords[i + 1, 0]],
                [path_coords[i, 1], path_coords[i + 1, 1]],
                color=color, linewidth=3, alpha=0.9, zorder=10)

    # Plot all CPs (faded)
    for cp_type in [0, 1, 2]:
        mask = types == cp_type
        ax3.scatter(coords[mask, 0], coords[mask, 1],
                   c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                   s=30, alpha=0.3, edgecolors='none', zorder=3)

    # Plot all VCs (faded)
    ax3.scatter(vc_coords[:, 0], vc_coords[:, 1],
               c='orange', marker='*', s=60, alpha=0.3,
               edgecolors='none', zorder=3)

    # Highlight source CP
    ax3.scatter(coords[source_cp, 0], coords[source_cp, 1],
               c='cyan', marker='o', s=200,
               edgecolors='black', linewidths=2, zorder=15, label='Source CP')

    # Highlight intermediate VCs on the path
    intermediate_vcs = [node for node in path if node >= n_cp and node != target_vc_node]
    for vc_node in intermediate_vcs:
        vc_idx = vc_node - n_cp
        ax3.scatter(vc_coords[vc_idx, 0], vc_coords[vc_idx, 1],
                   c='yellow', marker='*', s=250,
                   edgecolors='black', linewidths=1.5, zorder=14)
        # Add label for intermediate VC
        ax3.annotate(f'VC{vc_idx + 1}',
                    xy=(vc_coords[vc_idx, 0], vc_coords[vc_idx, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold', color='darkblue')

    # Highlight target VC
    ax3.scatter(vc_coords[target_region, 0], vc_coords[target_region, 1],
               c='orange', marker='*', s=400,
               edgecolors='black', linewidths=2, zorder=15, label='Target VC')

    # Add label for target VC
    ax3.annotate(f'VC{target_region + 1}',
                xy=(vc_coords[target_region, 0], vc_coords[target_region, 1]),
                xytext=(5, -12), textcoords='offset points',
                fontsize=9, fontweight='bold', color='darkorange')

    # Build path string
    path_str = ' → '.join(path_labels)

    # Create legend for path edges
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan',
               markersize=10, markeredgecolor='black', label='Source CP'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow',
               markersize=10, markeredgecolor='black', label='Intermediate VC'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='orange',
               markersize=12, markeredgecolor='black', label='Target VC'),
        Line2D([0], [0], color='blue', linewidth=3, label='CP↔CP edge'),
        Line2D([0], [0], color='green', linewidth=3, label='CP↔VC edge'),
        Line2D([0], [0], color='red', linewidth=3, label='VC↔VC edge'),
    ]

    ax3.set_xlim(-5, 105)
    ax3.set_ylim(-5, 105)
    ax3.set_xlabel('X', fontsize=11)
    ax3.set_ylabel('Y', fontsize=11)
    ax3.set_title(f'Shortest Path: CP {source_cp} → Region {target_region + 1}\n'
                 f'Path: {path_str}\nω({source_cp}, {target_region + 1}) = {path_dist:.2f}',
                 fontsize=11, fontweight='bold')
    ax3.legend(handles=legend_elements, loc='lower center', fontsize=8,
               framealpha=0.95, ncol=3, bbox_to_anchor=(0.5, -0.18))
    ax3.set_aspect('equal')

    # =========================================================================
    # Save figure
    # =========================================================================
    plt.suptitle(f'ω (Hypernetwork Function): Augmented Hypergraph with VC-VC Adjacency ({prefix.capitalize()})',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {save_path}")

    return {
        'n_cp': n_cp,
        'n_vc': n_hyper,
        'n_cp_cp_edges': len(cp_cp_edges),
        'n_cp_vc_edges': len(cp_vc_edges),
        'n_vc_vc_edges': len(vc_vc_edges),
        'n_vc_vc_strong': len(vc_vc_edges_strong)
    }


def main():
    print("=" * 60)
    print("VISUALIZING VC ADJACENCY FOR AUGMENTED HYPERGRAPH")
    print("=" * 60)

    # Process clean
    print("\nProcessing clean MS complex...")
    clean_path = os.path.join(BASE_PATH, "clean_vc_adjacency.png")
    clean_stats = plot_vc_adjacency("clean", clean_path)
    print(f"  CPs: {clean_stats['n_cp']}, VCs: {clean_stats['n_vc']}")
    print(f"  Edges: CP↔CP={clean_stats['n_cp_cp_edges']}, "
          f"CP↔VC={clean_stats['n_cp_vc_edges']}, "
          f"VC↔VC={clean_stats['n_vc_vc_edges']} (strong: {clean_stats['n_vc_vc_strong']})")

    # Process noisy
    print("\nProcessing noisy MS complex...")
    noisy_path = os.path.join(BASE_PATH, "noisy_vc_adjacency.png")
    noisy_stats = plot_vc_adjacency("noisy", noisy_path)
    print(f"  CPs: {noisy_stats['n_cp']}, VCs: {noisy_stats['n_vc']}")
    print(f"  Edges: CP↔CP={noisy_stats['n_cp_cp_edges']}, "
          f"CP↔VC={noisy_stats['n_cp_vc_edges']}, "
          f"VC↔VC={noisy_stats['n_vc_vc_edges']} (strong: {noisy_stats['n_vc_vc_strong']})")

    print("\n" + "=" * 60)
    print("Output files:")
    print(f"  - {clean_path}")
    print(f"  - {noisy_path}")
    print("=" * 60)
    print("DONE!")


if __name__ == "__main__":
    main()
