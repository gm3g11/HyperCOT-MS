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

    # 3. VC ↔ VC edges (adjacent regions - only strong edges with 2+ shared CPs)
    # This represents true region adjacency (sharing a separatrix), not just corner contact
    hyperedge_members = [eval(row['hyperedge']) for _, row in hyper_df.iterrows()]
    for i in range(n_hyper):
        members_i = set(hyperedge_members[i])
        for j in range(i + 1, n_hyper):
            members_j = set(hyperedge_members[j])
            shared = members_i & members_j
            if len(shared) >= 2:  # Only strong edges (4-neighbor adjacency)
                vc_i = n_cp + i
                vc_j = n_cp + j
                dist = np.linalg.norm(vc_coords[i] - vc_coords[j])
                adj[vc_i, vc_j] = dist
                adj[vc_j, vc_i] = dist
                vc_vc_edges.append((i, j))

    return adj, cp_cp_edges, cp_vc_edges, vc_vc_edges, n_cp, n_hyper


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

    # Build augmented graph (VC-VC edges use only 2+ shared CPs)
    adj, cp_cp_edges, cp_vc_edges, vc_vc_edges, n_cp, n_hyper = build_augmented_graph(
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

    # Draw ALL regions as faded polygons first
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    region_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    patches = []
    colors = []

    for region_idx, r_row in hyper_df.iterrows():
        r_min_id = int(r_row['min_id'])
        r_max_id = int(r_row['max_id'])
        r_saddle_ids = eval(r_row['boundary_saddles'])

        if len(r_saddle_ids) >= 2:
            r_vertices = np.array([
                coords[r_min_id],
                coords[r_saddle_ids[0]],
                coords[r_max_id],
                coords[r_saddle_ids[1]]
            ])
            # Sort by angle
            r_cx, r_cy = np.mean(r_vertices[:, 0]), np.mean(r_vertices[:, 1])
            r_angles = np.arctan2(r_vertices[:, 1] - r_cy, r_vertices[:, 0] - r_cx)
            r_sorted_idx = np.argsort(r_angles)
            r_vertices = r_vertices[r_sorted_idx]

            polygon = Polygon(r_vertices, closed=True)
            patches.append(polygon)
            colors.append(region_colors[region_idx % 20])

    # Add all region patches (faded)
    p = PatchCollection(patches, alpha=0.25, zorder=1)
    p.set_facecolors(colors)
    p.set_edgecolors('gray')
    p.set_linewidth(0.5)
    ax1.add_collection(p)

    # Plot all VCs (faded)
    ax1.scatter(vc_coords[:, 0], vc_coords[:, 1],
               c='orange', marker='*', s=60, alpha=0.3,
               edgecolors='none', zorder=2)

    # Plot all CPs (faded)
    for cp_type in [0, 1, 2]:
        mask = types == cp_type
        ax1.scatter(coords[mask, 0], coords[mask, 1],
                   c=CP_COLORS[cp_type], marker=CP_MARKERS[cp_type],
                   s=30, alpha=0.3, edgecolors='none', zorder=3)

    # Highlight the example region as a brighter polygon
    region_vertices = np.array([
        [min_pos[0], min_pos[1]],
        [s1_pos[0], s1_pos[1]],
        [max_pos[0], max_pos[1]],
        [s2_pos[0], s2_pos[1]]
    ])
    cx, cy = np.mean(region_vertices[:, 0]), np.mean(region_vertices[:, 1])
    angles = np.arctan2(region_vertices[:, 1] - cy, region_vertices[:, 0] - cx)
    sorted_idx = np.argsort(angles)
    region_vertices = region_vertices[sorted_idx]

    region_poly = Polygon(region_vertices, closed=True,
                          facecolor='yellow', edgecolor='darkorange',
                          alpha=0.5, linewidth=2.5, zorder=6)
    ax1.add_patch(region_poly)

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

    # Plot highlighted virtual center (intersection)
    ax1.scatter(vc_pos[0], vc_pos[1], c='orange', marker='*', s=400,
               edgecolors='black', linewidths=2, zorder=15, label='Virtual Center')

    # Add annotation
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

    # Custom legend with 3 columns x 2 rows
    from matplotlib.patches import Patch
    legend_elements_p1 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CP_COLORS[0],
               markersize=10, markeredgecolor='black', label='Minimum'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CP_COLORS[1],
               markersize=10, markeredgecolor='black', label='Saddle'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CP_COLORS[2],
               markersize=10, markeredgecolor='black', label='Maximum'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='Min-Max line'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Saddle-Saddle line'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='orange',
               markersize=12, markeredgecolor='black', label='Virtual Center'),
    ]
    ax1.legend(handles=legend_elements_p1, loc='lower center', fontsize=8,
               framealpha=0.95, ncol=3, bbox_to_anchor=(0.5, -0.18))
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
                color='gray', linewidth=0.8, alpha=0.4, zorder=1)

    # Draw VC↔VC edges (2+ shared CPs = true adjacency), more subtle
    for i, j in vc_vc_edges:
        ax2.plot([vc_coords[i, 0], vc_coords[j, 0]],
                [vc_coords[i, 1], vc_coords[j, 1]],
                color='orange', linewidth=1.0, alpha=0.35, zorder=2)

    # Draw CP↔VC edges - more visible now
    for cp_idx, vc_idx in cp_vc_edges:
        ax2.plot([coords[cp_idx, 0], vc_coords[vc_idx, 0]],
                [coords[cp_idx, 1], vc_coords[vc_idx, 1]],
                color='purple', linewidth=0.8, alpha=0.5, zorder=3)

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
        Line2D([0], [0], color='purple', linewidth=1.5, label='CP↔VC (boundary)'),
        Line2D([0], [0], color='orange', linewidth=2, label='VC↔VC (adjacent)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='orange',
               markersize=10, label='Virtual Center'),
    ]

    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-5, 105)
    ax2.set_xlabel('X', fontsize=11)
    ax2.set_ylabel('Y', fontsize=11)
    ax2.set_title(f'Augmented Graph\n({n_cp} CPs + {n_hyper} VCs, {len(vc_vc_edges)} VC-VC edges)',
                 fontsize=11, fontweight='bold')
    ax2.legend(handles=legend_elements, loc='lower center', fontsize=8,
               framealpha=0.95, ncol=2, bbox_to_anchor=(0.5, -0.18))
    ax2.set_aspect('equal')

    # =========================================================================
    # Panel 3: Example Shortest Path
    # =========================================================================
    ax3 = axes[2]

    # Use specific examples: middle CP to bottom region
    # Find CP closest to center (middle of domain)
    center = np.array([50, 50])
    cp_distances_to_center = np.linalg.norm(coords - center, axis=1)
    source_cp = np.argmin(cp_distances_to_center)

    # Find region with VC closest to bottom (lowest y value)
    target_region = np.argmin(vc_coords[:, 1])

    target_vc_node = n_cp + target_region

    # Find shortest path
    path, path_dist = find_shortest_path(adj, source_cp, target_vc_node, n_cp)

    # Draw all edges (faded background) - same as middle panel
    for i, j in cp_cp_edges:
        ax3.plot([coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color='gray', linewidth=0.5, alpha=0.2, zorder=1)

    # Draw VC-VC edges (2+ shared CPs) - same as middle panel
    for i, j in vc_vc_edges:
        ax3.plot([vc_coords[i, 0], vc_coords[j, 0]],
                [vc_coords[i, 1], vc_coords[j, 1]],
                color='orange', linewidth=0.6, alpha=0.2, zorder=1)

    # CP↔VC edges more visible
    for cp_idx, vc_idx in cp_vc_edges:
        ax3.plot([coords[cp_idx, 0], vc_coords[vc_idx, 0]],
                [coords[cp_idx, 1], vc_coords[vc_idx, 1]],
                color='purple', linewidth=0.6, alpha=0.35, zorder=2)

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

    # Highlight target region as shaded polygon
    target_row = hyper_df.iloc[target_region]
    target_min_id = int(target_row['min_id'])
    target_max_id = int(target_row['max_id'])
    target_saddle_ids = eval(target_row['boundary_saddles'])

    target_region_vertices = np.array([
        coords[target_min_id],
        coords[target_saddle_ids[0]],
        coords[target_max_id],
        coords[target_saddle_ids[1]]
    ])
    # Sort by angle for proper polygon
    tcx, tcy = np.mean(target_region_vertices[:, 0]), np.mean(target_region_vertices[:, 1])
    t_angles = np.arctan2(target_region_vertices[:, 1] - tcy, target_region_vertices[:, 0] - tcx)
    t_sorted_idx = np.argsort(t_angles)
    target_region_vertices = target_region_vertices[t_sorted_idx]

    from matplotlib.patches import Polygon
    target_poly = Polygon(target_region_vertices, closed=True,
                          facecolor='lightyellow', edgecolor='orange',
                          alpha=0.5, linewidth=2, zorder=4)
    ax3.add_patch(target_poly)

    # Highlight source CP
    ax3.scatter(coords[source_cp, 0], coords[source_cp, 1],
               c='cyan', marker='o', s=250,
               edgecolors='black', linewidths=2, zorder=15)

    # Add label for source CP
    ax3.annotate(f'CP {source_cp}',
                xy=(coords[source_cp, 0], coords[source_cp, 1]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=10, fontweight='bold', color='darkblue')

    # Highlight target VC
    ax3.scatter(vc_coords[target_region, 0], vc_coords[target_region, 1],
               c='orange', marker='*', s=400,
               edgecolors='black', linewidths=2, zorder=15)

    # Add label for target region
    ax3.annotate(f'Region {target_region + 1}',
                xy=(vc_coords[target_region, 0], vc_coords[target_region, 1]),
                xytext=(5, -15), textcoords='offset points',
                fontsize=10, fontweight='bold', color='darkorange')

    # Build path string
    path_str = ' → '.join(path_labels)

    # Create legend for path edges
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan',
               markersize=10, markeredgecolor='black', label='Source CP'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='orange',
               markersize=12, markeredgecolor='black', label='Target Region'),
        Line2D([0], [0], color='blue', linewidth=3, label='CP↔CP edge'),
        Line2D([0], [0], color='green', linewidth=3, label='CP↔VC edge'),
        Line2D([0], [0], color='red', linewidth=3, label='VC↔VC edge'),
    ]

    ax3.set_xlim(-5, 105)
    ax3.set_ylim(-5, 105)
    ax3.set_xlabel('X', fontsize=11)
    ax3.set_ylabel('Y', fontsize=11)
    ax3.set_title(f'Shortest Path Example\nω(CP {source_cp}, Region {target_region + 1}) = {path_dist:.2f}',
                 fontsize=11, fontweight='bold')
    ax3.legend(handles=legend_elements, loc='lower center', fontsize=8,
               framealpha=0.95, ncol=3, bbox_to_anchor=(0.5, -0.18))
    ax3.set_aspect('equal')

    # =========================================================================
    # Save figure
    # =========================================================================
    plt.suptitle(f'Hypernetwork Function ω: Distance from CP to Region ({prefix.capitalize()} MS Complex)',
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
        'n_vc_vc_edges': len(vc_vc_edges)
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
          f"VC↔VC={clean_stats['n_vc_vc_edges']} (2+ shared CPs)")

    # Process noisy
    print("\nProcessing noisy MS complex...")
    noisy_path = os.path.join(BASE_PATH, "noisy_vc_adjacency.png")
    noisy_stats = plot_vc_adjacency("noisy", noisy_path)
    print(f"  CPs: {noisy_stats['n_cp']}, VCs: {noisy_stats['n_vc']}")
    print(f"  Edges: CP↔CP={noisy_stats['n_cp_cp_edges']}, "
          f"CP↔VC={noisy_stats['n_cp_vc_edges']}, "
          f"VC↔VC={noisy_stats['n_vc_vc_edges']} (2+ shared CPs)")

    print("\n" + "=" * 60)
    print("Output files:")
    print(f"  - {clean_path}")
    print(f"  - {noisy_path}")
    print("=" * 60)
    print("DONE!")


if __name__ == "__main__":
    main()
