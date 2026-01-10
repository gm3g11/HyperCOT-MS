"""
Compute Node Measure (μ) for MS Complex using Extended Persistence + Persistence Image

This script:
1. Loads MS complex data (critical points + edges)
2. Computes Extended Persistence using gudhi
3. Generates Persistence Image
4. Computes node measure μ for each critical point
5. Saves results to CSV
"""

import numpy as np
import pandas as pd
import gudhi as gd
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='gudhi')

BASE_PATH = Path(__file__).parent.resolve()

# =============================================================================
# DATA LOADING
# =============================================================================

def load_ms_complex(prefix):
    """Load MS complex data: critical points and edges."""
    cp_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_critical_points.csv"))
    sep_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_separatrices_cells.csv"))

    # Extract CP info
    cp_data = pd.DataFrame({
        'point_id': cp_df['Point ID'],
        'cell_id': cp_df['CellId'],
        'cp_type': cp_df['CellDimension'],
        'x': cp_df['Points_0'],
        'y': cp_df['Points_1'],
        'z': cp_df['Points_2'],  # Use z as filtration value
    })

    # Build edge list from separatrices
    cell_id_to_idx = {cid: idx for idx, cid in enumerate(cp_data['cell_id'].values)}

    unique_sep = sep_df.drop_duplicates(subset=['SeparatrixId'])[
        ['SeparatrixId', 'SourceId', 'DestinationId']
    ]

    edges = []
    for _, row in unique_sep.iterrows():
        src = int(row['SourceId'])
        dst = int(row['DestinationId'])
        if src in cell_id_to_idx and dst in cell_id_to_idx:
            i = cell_id_to_idx[src]
            j = cell_id_to_idx[dst]
            edges.append((i, j))

    return cp_data, edges


# =============================================================================
# EXTENDED PERSISTENCE
# =============================================================================

def compute_extended_persistence(cp_data, edges):
    """
    Compute extended persistence using gudhi.

    Args:
        cp_data: DataFrame with 'z' column as filtration values
        edges: List of (i, j) edge tuples

    Returns:
        Dictionary with persistence diagrams by type
    """
    n_cp = len(cp_data)
    z_values = cp_data['z'].values

    # Build simplex tree
    st = gd.SimplexTree()

    # Add vertices (0-simplices) with filtration = z value
    for i in range(n_cp):
        st.insert([i], filtration=z_values[i])

    # Add edges (1-simplices) with filtration = max of endpoints
    for i, j in edges:
        f_edge = max(z_values[i], z_values[j])
        st.insert([i, j], filtration=f_edge)

    # Compute extended persistence
    st.make_filtration_non_decreasing()
    st.extend_filtration()

    dgms = st.extended_persistence()

    # Parse diagrams by type
    types = ["Ordinary", "Relative", "Extended+", "Extended-"]
    pairs_by_type = {t: [] for t in types}
    all_pairs_bd = []  # (birth, death) pairs

    tolerance = 1e-5

    for t_idx, dgm in enumerate(dgms):
        tname = types[t_idx]
        for dim, (b, d) in dgm:
            if np.isfinite(b) and np.isfinite(d) and abs(b - d) > tolerance:
                pairs_by_type[tname].append((b, d))
                all_pairs_bd.append((b, d))

    return {
        'pairs_by_type': pairs_by_type,
        'all_pairs_bd': all_pairs_bd,
        'z_values': z_values
    }


# =============================================================================
# PERSISTENCE IMAGE
# =============================================================================

def compute_persistence_image(all_pairs_bd, nx_res=100, ny_res=100, sigma=3.0, weight_power=1.0):
    """
    Compute Persistence Image from persistence pairs.

    Args:
        all_pairs_bd: List of (birth, death) pairs
        nx_res, ny_res: Grid resolution
        sigma: Gaussian kernel bandwidth
        weight_power: Power for persistence weighting

    Returns:
        Dictionary with PI data
    """
    if not all_pairs_bd:
        return {
            'persistence_image': np.zeros(nx_res * ny_res),
            'finite_pairs_bp': [],
            'birth_range': np.linspace(0, 1, nx_res),
            'persistence_range': np.linspace(0, 1, ny_res),
            'grid_centers': None
        }

    # Transform to (birth, persistence) coordinates
    tolerance = 1e-5
    finite_pairs_bp = [(b, abs(d - b)) for b, d in all_pairs_bd if abs(d - b) > tolerance]

    if not finite_pairs_bp:
        return {
            'persistence_image': np.zeros(nx_res * ny_res),
            'finite_pairs_bp': [],
            'birth_range': np.linspace(0, 1, nx_res),
            'persistence_range': np.linspace(0, 1, ny_res),
            'grid_centers': None
        }

    # Compute weights (persistence^weight_power)
    weights = [p**weight_power for _, p in finite_pairs_bp]

    # Define grid bounds
    bs = [b for b, _ in finite_pairs_bp]
    ps = [p for _, p in finite_pairs_bp]
    min_b, max_b = min(bs), max(bs)
    min_p, max_p = min(ps), max(ps)

    # Add buffer around bounds
    buf_b = 3 * sigma + 0.05 * (max_b - min_b) if max_b > min_b else 3 * sigma
    buf_p = 3 * sigma + 0.05 * (max_p - min_p) if max_p > min_p else 3 * sigma

    birth_range = np.linspace(min_b - buf_b, max_b + buf_b, nx_res)
    persistence_range = np.linspace(max(0, min_p - buf_p), max_p + buf_p, ny_res)

    X, Y = np.meshgrid(birth_range, persistence_range)
    grid_centers = np.vstack([X.ravel(), Y.ravel()]).T

    # Compute PI via weighted Gaussians
    I_f = np.zeros(nx_res * ny_res)
    for (b_val, p_val), w in zip(finite_pairs_bp, weights):
        gauss = multivariate_normal(mean=(b_val, p_val), cov=(sigma**2) * np.eye(2))
        I_f += w * gauss.pdf(grid_centers)

    return {
        'persistence_image': I_f,
        'finite_pairs_bp': finite_pairs_bp,
        'birth_range': birth_range,
        'persistence_range': persistence_range,
        'grid_centers': grid_centers,
        'nx_res': nx_res,
        'ny_res': ny_res
    }


# =============================================================================
# NODE MEASURE (μ)
# =============================================================================

def compute_node_measure(cp_data, persistence_result, pi_result, sigma=3.0, tolerance=1e-5):
    """
    Compute node measure μ for each critical point.

    Each CP is mapped to a (birth, persistence) point based on the extended persistence.
    The contribution is computed as the overlap of the CP's Gaussian with the PI.

    Args:
        cp_data: DataFrame with CP info
        persistence_result: Output from compute_extended_persistence
        pi_result: Output from compute_persistence_image
        sigma: Gaussian bandwidth
        tolerance: Tolerance for matching f-values

    Returns:
        Dictionary with node measures and mappings
    """
    n_cp = len(cp_data)
    z_values = persistence_result['z_values']
    all_pairs_bd = persistence_result['all_pairs_bd']
    I_f = pi_result['persistence_image']
    grid_centers = pi_result['grid_centers']

    if grid_centers is None or len(all_pairs_bd) == 0:
        # No persistence pairs - return uniform measure
        print("  Warning: No persistence pairs found. Using uniform measure.")
        mu = np.ones(n_cp) / n_cp
        return {
            'mu': mu,
            'vertex_to_bp_map': {},
            'contributions': {}
        }

    # Step 1: Heuristic mapping of vertices to (birth, persistence) points
    # Each vertex corresponds to either a birth or death in the persistence diagram
    vertex_to_bp_map = {}

    # Group vertices by z-value for matching
    value_to_vertices = {}
    for i, z_val in enumerate(z_values):
        if z_val not in value_to_vertices:
            value_to_vertices[z_val] = []
        value_to_vertices[z_val].append(i)

    possible_z_values = np.array(list(value_to_vertices.keys()))

    # Map each persistence pair to vertices
    for b, d in all_pairs_bd:
        p = abs(d - b)
        point_bp = (b, p)

        # Find vertices with z-value matching birth
        birth_match_indices = np.where(np.isclose(possible_z_values, b, atol=tolerance))[0]
        birth_vertices = [v for idx in birth_match_indices
                         for v in value_to_vertices[possible_z_values[idx]]]

        # Find vertices with z-value matching death
        death_match_indices = np.where(np.isclose(possible_z_values, d, atol=tolerance))[0]
        death_vertices = [v for idx in death_match_indices
                         for v in value_to_vertices[possible_z_values[idx]]]

        # Map matched vertices to this (birth, persistence) point
        for v in birth_vertices + death_vertices:
            if v not in vertex_to_bp_map:
                vertex_to_bp_map[v] = point_bp

    # Step 2: Compute contributions
    contributions = {}

    for v in range(n_cp):
        if v in vertex_to_bp_map:
            b_v, p_v = vertex_to_bp_map[v]
            gauss_v = multivariate_normal(mean=[b_v, p_v], cov=(sigma**2) * np.eye(2))
            contributions[v] = np.sum(I_f * gauss_v.pdf(grid_centers))
        else:
            # Vertex not mapped - assign small contribution
            contributions[v] = 0.0

    # Step 3: Normalize to probability measure
    total_contribution = sum(contributions.values())

    if total_contribution > 1e-9:
        mu = np.array([contributions[v] / total_contribution for v in range(n_cp)])
    else:
        print("  Warning: Total contribution near zero. Using uniform measure.")
        mu = np.ones(n_cp) / n_cp

    return {
        'mu': mu,
        'vertex_to_bp_map': vertex_to_bp_map,
        'contributions': contributions
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_persistence_results(prefix, cp_data, persistence_result, pi_result, mu_result, save_path):
    """Plot persistence diagram, persistence image, and μ distribution."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Persistence Diagram
    ax1 = axes[0]
    pairs_by_type = persistence_result['pairs_by_type']
    markers = {'Ordinary': 'o', 'Relative': 's', 'Extended+': '^', 'Extended-': 'x'}
    colors = {'Ordinary': 'blue', 'Relative': 'red', 'Extended+': 'green', 'Extended-': 'purple'}

    all_pts = []
    for t, pairs in pairs_by_type.items():
        if pairs:
            arr = np.array(pairs)
            all_pts.append(arr)
            ax1.scatter(arr[:, 0], arr[:, 1], label=f'{t} ({len(pairs)})',
                       marker=markers[t], color=colors[t], s=50, alpha=0.8)

    if all_pts:
        M = np.vstack(all_pts)
        mn, mx = M.min(), M.max()
        buf = 0.05 * (mx - mn) + 1
        ax1.plot([mn - buf, mx + buf], [mn - buf, mx + buf], 'k--', linewidth=1)
        ax1.set_xlim(mn - buf, mx + buf)
        ax1.set_ylim(mn - buf, mx + buf)

    ax1.set_xlabel('Birth')
    ax1.set_ylabel('Death')
    ax1.set_title(f'{prefix}: Extended Persistence Diagram')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=':')

    # 2. Persistence Image
    ax2 = axes[1]
    if pi_result['persistence_image'].sum() > 0:
        I2 = pi_result['persistence_image'].reshape(pi_result['ny_res'], pi_result['nx_res'])
        im = ax2.imshow(I2, origin='lower',
                       extent=[pi_result['birth_range'][0], pi_result['birth_range'][-1],
                              pi_result['persistence_range'][0], pi_result['persistence_range'][-1]],
                       aspect='auto', cmap='viridis')
        fig.colorbar(im, ax=ax2, label='Intensity')

        # Overlay (birth, persistence) points
        if pi_result['finite_pairs_bp']:
            bp_arr = np.array(pi_result['finite_pairs_bp'])
            ax2.scatter(bp_arr[:, 0], bp_arr[:, 1], c='red', s=30,
                       edgecolors='white', linewidths=0.5, label='(b, p) points')
            ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No finite pairs', ha='center', va='center', transform=ax2.transAxes)

    ax2.set_xlabel('Birth')
    ax2.set_ylabel('Persistence')
    ax2.set_title(f'{prefix}: Persistence Image')

    # 3. μ Distribution
    ax3 = axes[2]
    mu = mu_result['mu']
    cp_types = cp_data['cp_type'].values

    # Color by CP type
    type_colors = {0: 'blue', 1: 'green', 2: 'red'}
    type_labels = {0: 'Min', 1: 'Saddle', 2: 'Max'}

    for t in [0, 1, 2]:
        mask = cp_types == t
        indices = np.where(mask)[0]
        ax3.bar(indices, mu[mask], color=type_colors[t], label=type_labels[t], alpha=0.7)

    ax3.set_xlabel('CP Index')
    ax3.set_ylabel('μ (Probability)')
    ax3.set_title(f'{prefix}: Node Measure μ')
    ax3.legend(fontsize=8)
    ax3.grid(True, axis='y', linestyle=':')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def process_ms_complex(prefix, sigma=3.0, nx_res=100, ny_res=100):
    """Process a single MS complex: compute extended persistence, PI, and μ."""

    print(f"\n{'='*60}")
    print(f"Processing {prefix} MS complex")
    print(f"{'='*60}")

    # Load data
    print("\n1. Loading data...")
    cp_data, edges = load_ms_complex(prefix)
    print(f"   CPs: {len(cp_data)}")
    print(f"   Edges: {len(edges)}")
    print(f"   Z range: [{cp_data['z'].min():.2f}, {cp_data['z'].max():.2f}]")

    # Compute extended persistence
    print("\n2. Computing extended persistence...")
    persistence_result = compute_extended_persistence(cp_data, edges)

    for t, pairs in persistence_result['pairs_by_type'].items():
        if pairs:
            print(f"   {t}: {len(pairs)} pairs")
    print(f"   Total finite pairs: {len(persistence_result['all_pairs_bd'])}")

    # Compute persistence image
    print("\n3. Computing persistence image...")
    pi_result = compute_persistence_image(
        persistence_result['all_pairs_bd'],
        nx_res=nx_res, ny_res=ny_res, sigma=sigma
    )
    print(f"   PI shape: ({ny_res}, {nx_res})")
    print(f"   PI sum: {pi_result['persistence_image'].sum():.4f}")

    # Compute node measure
    print("\n4. Computing node measure μ...")
    mu_result = compute_node_measure(cp_data, persistence_result, pi_result, sigma=sigma)

    mapped_count = len(mu_result['vertex_to_bp_map'])
    print(f"   Mapped vertices: {mapped_count}/{len(cp_data)}")
    print(f"   μ sum: {mu_result['mu'].sum():.6f}")
    print(f"   μ range: [{mu_result['mu'].min():.6f}, {mu_result['mu'].max():.6f}]")

    # Plot results
    print("\n5. Generating visualization...")
    plot_path = os.path.join(BASE_PATH, f"{prefix}_persistence_mu.png")
    plot_persistence_results(prefix, cp_data, persistence_result, pi_result, mu_result, plot_path)

    # Save μ to CSV
    print("\n6. Saving results...")
    mu_df = pd.DataFrame({
        'cp_index': range(len(cp_data)),
        'point_id': cp_data['point_id'].values,
        'cp_type': cp_data['cp_type'].values,
        'x': cp_data['x'].values,
        'y': cp_data['y'].values,
        'z': cp_data['z'].values,
        'mu': mu_result['mu']
    })

    csv_path = os.path.join(BASE_PATH, f"{prefix}_node_measure.csv")
    mu_df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    return {
        'cp_data': cp_data,
        'edges': edges,
        'persistence_result': persistence_result,
        'pi_result': pi_result,
        'mu_result': mu_result
    }


def main():
    print("="*60)
    print("COMPUTING NODE MEASURE (μ) FOR MS COMPLEXES")
    print("Using Extended Persistence + Persistence Image")
    print("="*60)

    # Parameters
    sigma = 3.0  # Gaussian bandwidth
    nx_res, ny_res = 100, 100  # PI resolution

    print(f"\nParameters:")
    print(f"  sigma (Gaussian bandwidth): {sigma}")
    print(f"  PI resolution: {nx_res} x {ny_res}")

    # Process clean MS complex
    clean_results = process_ms_complex("clean", sigma=sigma, nx_res=nx_res, ny_res=ny_res)

    # Process noisy MS complex
    noisy_results = process_ms_complex("noisy", sigma=sigma, nx_res=nx_res, ny_res=ny_res)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nClean MS Complex:")
    print(f"  CPs: {len(clean_results['cp_data'])}")
    print(f"  μ entropy: {-np.sum(clean_results['mu_result']['mu'] * np.log(clean_results['mu_result']['mu'] + 1e-10)):.4f}")

    print("\nNoisy MS Complex:")
    print(f"  CPs: {len(noisy_results['cp_data'])}")
    print(f"  μ entropy: {-np.sum(noisy_results['mu_result']['mu'] * np.log(noisy_results['mu_result']['mu'] + 1e-10)):.4f}")

    print("\nOutput files:")
    print("  - clean_node_measure.csv")
    print("  - noisy_node_measure.csv")
    print("  - clean_persistence_mu.png")
    print("  - noisy_persistence_mu.png")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

    return clean_results, noisy_results


if __name__ == "__main__":
    clean_results, noisy_results = main()
