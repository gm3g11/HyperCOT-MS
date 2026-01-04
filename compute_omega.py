"""
Compute ω (Hypernetwork Function) for MS Complex

ω(v, e) = shortest path distance from CP v to virtual center of hyperedge e
using augmented graph with VC-VC adjacency.

Graph structure:
- Nodes: CPs + VCs (virtual centers)
- Edges:
  1. CP ↔ CP: separatrices (L2 weight)
  2. CP ↔ VC: boundary CPs connect to their region's VC (L2 weight)
  3. VC ↔ VC: adjacent regions connected (L2 weight)
"""

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import os

BASE_PATH = "/Users/gmeng/Desktop/COOT on Morse-Smale"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ms_complex(prefix):
    """Load MS complex data: critical points, edges, and hypergraph."""
    cp_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_critical_points.csv"))
    sep_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_separatrices_cells.csv"))
    hyper_df = pd.read_csv(os.path.join(BASE_PATH, f"hypergraph_{prefix}.csv"))

    # Extract CP info
    cp_data = pd.DataFrame({
        'point_id': cp_df['Point ID'],
        'cell_id': cp_df['CellId'],
        'cp_type': cp_df['CellDimension'],
        'x': cp_df['Points_0'],
        'y': cp_df['Points_1'],
        'z': cp_df['Points_2'],
    })

    # Build edge list from separatrices
    cell_id_to_idx = {cid: idx for idx, cid in enumerate(cp_data['cell_id'].values)}
    unique_sep = sep_df.drop_duplicates(subset=['SeparatrixId'])[['SourceId', 'DestinationId']]

    edges = []
    for _, row in unique_sep.iterrows():
        src, dst = int(row['SourceId']), int(row['DestinationId'])
        if src in cell_id_to_idx and dst in cell_id_to_idx:
            edges.append((cell_id_to_idx[src], cell_id_to_idx[dst]))

    return cp_data, edges, hyper_df


# =============================================================================
# VIRTUAL CENTER COMPUTATION
# =============================================================================

def compute_virtual_center(cp_data, min_id, max_id, saddle_ids):
    """
    Compute virtual center as intersection of (min,max) and (saddle,saddle) lines.
    Uses 2D (x, y) intersection, then interpolates z.
    """
    min_pos = cp_data.iloc[min_id][['x', 'y', 'z']].values.astype(float)
    max_pos = cp_data.iloc[max_id][['x', 'y', 'z']].values.astype(float)
    s1_pos = cp_data.iloc[saddle_ids[0]][['x', 'y', 'z']].values.astype(float)
    s2_pos = cp_data.iloc[saddle_ids[1]][['x', 'y', 'z']].values.astype(float)

    # Line directions in 2D
    d1 = max_pos[:2] - min_pos[:2]
    d2 = s2_pos[:2] - s1_pos[:2]

    # Solve for intersection
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = np.array([s1_pos[0] - min_pos[0], s1_pos[1] - min_pos[1]])

    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        # Lines are parallel, use centroid
        center = (min_pos + max_pos + s1_pos + s2_pos) / 4
    else:
        params = np.linalg.solve(A, b)
        t, s = params[0], params[1]
        center_xy = min_pos[:2] + t * d1
        z1 = min_pos[2] + t * (max_pos[2] - min_pos[2])
        z2 = s1_pos[2] + s * (s2_pos[2] - s1_pos[2])
        center_z = (z1 + z2) / 2
        center = np.array([center_xy[0], center_xy[1], center_z])

    return center


def compute_all_virtual_centers(cp_data, hyper_df):
    """Compute virtual centers for all hyperedges."""
    virtual_centers = []
    hyperedge_members = []

    for _, row in hyper_df.iterrows():
        min_id = int(row['min_id'])
        max_id = int(row['max_id'])
        saddle_ids = eval(row['boundary_saddles'])

        vc = compute_virtual_center(cp_data, min_id, max_id, saddle_ids)
        virtual_centers.append(vc)

        members = [min_id] + list(saddle_ids) + [max_id]
        hyperedge_members.append(members)

    return np.array(virtual_centers), hyperedge_members


# =============================================================================
# AUGMENTED GRAPH CONSTRUCTION
# =============================================================================

def find_adjacent_regions(hyperedge_members):
    """Find pairs of adjacent regions (share at least one boundary CP)."""
    n_hyper = len(hyperedge_members)
    adjacency = []

    for i in range(n_hyper):
        members_i = set(hyperedge_members[i])
        for j in range(i + 1, n_hyper):
            members_j = set(hyperedge_members[j])
            shared = members_i & members_j
            if len(shared) > 0:
                adjacency.append((i, j, list(shared)))

    return adjacency


def build_augmented_graph(cp_data, edges, virtual_centers, hyperedge_members):
    """
    Build augmented graph with three edge types:
    1. CP ↔ CP: separatrices
    2. CP ↔ VC: boundary membership
    3. VC ↔ VC: adjacent regions
    """
    n_cp = len(cp_data)
    n_hyper = len(virtual_centers)
    n_total = n_cp + n_hyper

    coords = cp_data[['x', 'y', 'z']].values

    # Initialize adjacency matrix
    adj = np.full((n_total, n_total), np.inf)
    np.fill_diagonal(adj, 0)

    edge_counts = {'cp_cp': 0, 'cp_vc': 0, 'vc_vc': 0}

    # 1. CP ↔ CP edges (separatrices)
    for i, j in edges:
        dist = np.linalg.norm(coords[i] - coords[j])
        adj[i, j] = dist
        adj[j, i] = dist
        edge_counts['cp_cp'] += 1

    # 2. CP ↔ VC edges (boundary membership)
    for h_idx, members in enumerate(hyperedge_members):
        vc_node = n_cp + h_idx
        vc_pos = virtual_centers[h_idx]
        for cp_idx in members:
            dist = np.linalg.norm(coords[cp_idx] - vc_pos)
            adj[cp_idx, vc_node] = dist
            adj[vc_node, cp_idx] = dist
            edge_counts['cp_vc'] += 1

    # 3. VC ↔ VC edges (adjacent regions)
    vc_adjacency = find_adjacent_regions(hyperedge_members)
    for i, j, shared in vc_adjacency:
        vc_i = n_cp + i
        vc_j = n_cp + j
        dist = np.linalg.norm(virtual_centers[i] - virtual_centers[j])
        adj[vc_i, vc_j] = dist
        adj[vc_j, vc_i] = dist
        edge_counts['vc_vc'] += 1

    return adj, edge_counts, vc_adjacency


# =============================================================================
# ω COMPUTATION
# =============================================================================

def compute_omega(adj, n_cp, n_hyper):
    """
    Compute ω matrix using Dijkstra shortest paths.

    ω[v, e] = shortest path from CP v to virtual center of hyperedge e

    Returns:
        omega: Matrix of shape (n_cp, n_hyper)
    """
    # Compute all-pairs shortest paths from CPs to all nodes
    dist_matrix = dijkstra(csr_matrix(adj), directed=False, indices=range(n_cp))

    # Extract distances from CPs to VCs
    omega = dist_matrix[:, n_cp:n_cp + n_hyper]

    return omega


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_omega_results(prefix, cp_data, hyper_df, omega, virtual_centers, save_path):
    """Visualize ω matrix and statistics."""
    n_cp = len(cp_data)
    n_hyper = len(hyper_df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. ω matrix heatmap
    ax1 = axes[0]
    im = ax1.imshow(omega, aspect='auto', cmap='viridis')
    ax1.set_xlabel('Hyperedge (Region)')
    ax1.set_ylabel('Critical Point')
    ax1.set_title(f'{prefix}: ω Matrix\n(Shortest Path Distance)')
    fig.colorbar(im, ax=ax1, label='Distance')

    # 2. ω distribution
    ax2 = axes[1]
    omega_flat = omega.flatten()
    omega_finite = omega_flat[np.isfinite(omega_flat)]
    ax2.hist(omega_finite, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(omega_finite), color='red', linestyle='--',
                label=f'Mean: {np.mean(omega_finite):.2f}')
    ax2.axvline(np.median(omega_finite), color='orange', linestyle='--',
                label=f'Median: {np.median(omega_finite):.2f}')
    ax2.set_xlabel('ω (Distance)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{prefix}: ω Distribution')
    ax2.legend()

    # 3. Spatial visualization of ω for one hyperedge
    ax3 = axes[2]
    coords = cp_data[['x', 'y']].values
    types = cp_data['cp_type'].values

    # Choose a hyperedge to visualize
    target_hyper = n_hyper // 2
    omega_to_target = omega[:, target_hyper]

    # Color CPs by their ω to target hyperedge
    scatter = ax3.scatter(coords[:, 0], coords[:, 1], c=omega_to_target,
                         cmap='coolwarm', s=60, edgecolors='black', linewidths=0.5)
    fig.colorbar(scatter, ax=ax3, label=f'ω(·, Region {target_hyper+1})')

    # Mark target virtual center
    vc_pos = virtual_centers[target_hyper][:2]
    ax3.scatter(vc_pos[0], vc_pos[1], c='lime', marker='*', s=300,
               edgecolors='black', linewidths=2, label=f'VC (Region {target_hyper+1})')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title(f'{prefix}: ω to Region {target_hyper+1}')
    ax3.legend(loc='upper right')
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_ms_complex(prefix):
    """Process a single MS complex: compute ω."""
    print(f"\n{'='*60}")
    print(f"Processing {prefix} MS complex")
    print(f"{'='*60}")

    # Load data
    print("\n1. Loading data...")
    cp_data, edges, hyper_df = load_ms_complex(prefix)
    n_cp = len(cp_data)
    n_hyper = len(hyper_df)
    print(f"   CPs: {n_cp}")
    print(f"   Edges (separatrices): {len(edges)}")
    print(f"   Hyperedges (regions): {n_hyper}")

    # Compute virtual centers
    print("\n2. Computing virtual centers...")
    virtual_centers, hyperedge_members = compute_all_virtual_centers(cp_data, hyper_df)
    print(f"   Virtual centers: {len(virtual_centers)}")

    # Build augmented graph
    print("\n3. Building augmented graph...")
    adj, edge_counts, vc_adjacency = build_augmented_graph(
        cp_data, edges, virtual_centers, hyperedge_members
    )
    print(f"   Graph size: {adj.shape[0]} nodes")
    print(f"   CP↔CP edges: {edge_counts['cp_cp']}")
    print(f"   CP↔VC edges: {edge_counts['cp_vc']}")
    print(f"   VC↔VC edges: {edge_counts['vc_vc']}")
    print(f"   Total edges: {sum(edge_counts.values())}")

    # Compute ω
    print("\n4. Computing ω matrix...")
    omega = compute_omega(adj, n_cp, n_hyper)
    print(f"   ω shape: {omega.shape}")

    # Statistics
    omega_finite = omega[np.isfinite(omega)]
    print(f"   ω range: [{omega_finite.min():.2f}, {omega_finite.max():.2f}]")
    print(f"   ω mean: {omega_finite.mean():.2f}")
    print(f"   ω median: {np.median(omega_finite):.2f}")

    # Check for unreachable pairs
    n_inf = np.sum(~np.isfinite(omega))
    if n_inf > 0:
        print(f"   Warning: {n_inf} unreachable (CP, hyperedge) pairs")

    # Generate visualization
    print("\n5. Generating visualization...")
    plot_path = os.path.join(BASE_PATH, f"{prefix}_omega.png")
    plot_omega_results(prefix, cp_data, hyper_df, omega, virtual_centers, plot_path)

    # Save ω matrix to CSV
    print("\n6. Saving results...")

    # Save full ω matrix
    omega_df = pd.DataFrame(omega,
                            index=[f'CP_{i}' for i in range(n_cp)],
                            columns=[f'Region_{i+1}' for i in range(n_hyper)])
    omega_csv_path = os.path.join(BASE_PATH, f"{prefix}_omega.csv")
    omega_df.to_csv(omega_csv_path)
    print(f"   Saved: {omega_csv_path}")

    # Save virtual centers
    vc_df = pd.DataFrame({
        'region_id': range(1, n_hyper + 1),
        'vc_x': virtual_centers[:, 0],
        'vc_y': virtual_centers[:, 1],
        'vc_z': virtual_centers[:, 2],
        'boundary_cps': [str(m) for m in hyperedge_members]
    })
    vc_csv_path = os.path.join(BASE_PATH, f"{prefix}_virtual_centers.csv")
    vc_df.to_csv(vc_csv_path, index=False)
    print(f"   Saved: {vc_csv_path}")

    # Save VC adjacency
    adj_df = pd.DataFrame({
        'region_i': [a[0] + 1 for a in vc_adjacency],
        'region_j': [a[1] + 1 for a in vc_adjacency],
        'shared_cps': [str(a[2]) for a in vc_adjacency],
        'vc_distance': [np.linalg.norm(virtual_centers[a[0]] - virtual_centers[a[1]])
                       for a in vc_adjacency]
    })
    adj_csv_path = os.path.join(BASE_PATH, f"{prefix}_vc_adjacency.csv")
    adj_df.to_csv(adj_csv_path, index=False)
    print(f"   Saved: {adj_csv_path}")

    return {
        'cp_data': cp_data,
        'hyper_df': hyper_df,
        'virtual_centers': virtual_centers,
        'hyperedge_members': hyperedge_members,
        'omega': omega,
        'vc_adjacency': vc_adjacency
    }


def main():
    print("="*60)
    print("COMPUTING ω (HYPERNETWORK FUNCTION) FOR MS COMPLEXES")
    print("="*60)
    print("""
Method: Shortest path on augmented graph with VC-VC adjacency

Graph structure:
  - CP ↔ CP: separatrices (L2 weight)
  - CP ↔ VC: boundary CPs to their region's VC (L2 weight)
  - VC ↔ VC: adjacent regions (L2 weight)

Virtual center: intersection of (min,max) and (saddle,saddle) lines
""")

    # Process clean MS complex
    clean_results = process_ms_complex("clean")

    # Process noisy MS complex
    noisy_results = process_ms_complex("noisy")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nClean MS Complex:")
    print(f"  ω shape: {clean_results['omega'].shape}")
    print(f"  ω mean: {clean_results['omega'].mean():.2f}")
    print(f"  VC-VC adjacent pairs: {len(clean_results['vc_adjacency'])}")

    print("\nNoisy MS Complex:")
    print(f"  ω shape: {noisy_results['omega'].shape}")
    print(f"  ω mean: {noisy_results['omega'].mean():.2f}")
    print(f"  VC-VC adjacent pairs: {len(noisy_results['vc_adjacency'])}")

    print("\nOutput files:")
    print("  - clean_omega.csv, noisy_omega.csv (ω matrices)")
    print("  - clean_virtual_centers.csv, noisy_virtual_centers.csv")
    print("  - clean_vc_adjacency.csv, noisy_vc_adjacency.csv")
    print("  - clean_omega.png, noisy_omega.png (visualizations)")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

    return clean_results, noisy_results


if __name__ == "__main__":
    clean_results, noisy_results = main()
