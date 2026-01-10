"""
HyperCOT: Co-Optimal Transport for Hypergraphs

Computes the optimal coupling between two MS complex hypergraphs using POT's COOT.

Objective:
    min_{π, ξ} Σ_{v,v'} Σ_{e,e'} |ω₁(v,e) - ω₂(v',e')|² π(v,v') ξ(e,e')

Subject to:
    - π has marginals μ₁ and μ₂ (node measures)
    - ξ has marginals ν₁ and ν₂ (hyperedge measures)

Uses POT library's co_optimal_transport implementation.
Reference: Redko et al. (2020) "CO-Optimal Transport", NeurIPS.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ot
from ot import coot
import os
import warnings

warnings.filterwarnings('ignore')

BASE_PATH = "/Users/gmeng/Desktop/COOT on Morse-Smale"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_hypercot_data(prefix):
    """Load all HyperCOT components for one MS complex."""
    # Load μ (node measure)
    mu_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_node_measure.csv"))
    mu = mu_df['mu'].values

    # Load ν (hyperedge measure)
    nu_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_nu.csv"))
    nu = nu_df['nu'].values

    # Load ω (hypernetwork function)
    omega_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_omega.csv"), index_col=0)
    omega = omega_df.values

    # Load CP data for visualization
    cp_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_critical_points.csv"))
    cp_data = pd.DataFrame({
        'x': cp_df['Points_0'],
        'y': cp_df['Points_1'],
        'z': cp_df['Points_2'],
        'cp_type': cp_df['CellDimension']
    })

    # Load virtual centers
    vc_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_virtual_centers.csv"))

    return {
        'mu': mu,
        'nu': nu,
        'omega': omega,
        'cp_data': cp_data,
        'vc_data': vc_df
    }


# =============================================================================
# HYPERCOT USING POT LIBRARY
# =============================================================================

def run_coot(omega1, omega2, mu1, mu2, nu1, nu2,
             cp_types1=None, cp_types2=None, alpha_type=0.5,
             epsilon=0, nits_bcd=100, tol_bcd=1e-7,
             nits_ot=500, tol_sinkhorn=1e-7, verbose=True):
    """
    Run CO-Optimal Transport using POT library.

    COOT finds joint optimal coupling for samples (nodes) and features (hyperedges).

    In our MS complex context:
        - X = omega1 (n1 x m1): Clean hypernetwork function (nodes x hyperedges)
        - Y = omega2 (n2 x m2): Noisy hypernetwork function (nodes x hyperedges)
        - wx_samp = mu1, wy_samp = mu2: Node measures
        - wx_feat = nu1, wy_feat = nu2: Hyperedge measures

    Args:
        omega1, omega2: Hypernetwork functions (n1 x m1) and (n2 x m2)
        mu1, mu2: Node measures (samples)
        nu1, nu2: Hyperedge measures (features)
        cp_types1, cp_types2: CP types for type-based cost penalty
        alpha_type: Weight for type-based cost (0 = no penalty, higher = stronger)
        epsilon: Entropy regularization (0 = exact EMD, >0 = Sinkhorn)
        nits_bcd: Number of Block Coordinate Descent iterations
        tol_bcd: BCD convergence tolerance
        nits_ot: Number of OT iterations per BCD step
        tol_sinkhorn: Sinkhorn convergence tolerance
        verbose: Print progress

    Returns:
        pi: Node coupling (n1 x n2)
        xi: Hyperedge coupling (m1 x m2)
        log: Optimization log dictionary
    """
    n1, m1 = omega1.shape
    n2, m2 = omega2.shape

    if verbose:
        print(f"  Running POT COOT...")
        print(f"  Nodes: {n1} x {n2}, Hyperedges: {m1} x {m2}")
        print(f"  Epsilon: {epsilon} ({'EMD' if epsilon == 0 else 'Sinkhorn'})")

    # Normalize ω matrices for numerical stability
    scale = max(omega1.max(), omega2.max(), 1e-10)
    omega1_norm = omega1 / scale
    omega2_norm = omega2 / scale

    # Normalize marginals
    mu1_norm = mu1 / mu1.sum()
    mu2_norm = mu2 / mu2.sum()
    nu1_norm = nu1 / nu1.sum()
    nu2_norm = nu2 / nu2.sum()

    # Build type-based cost matrix if CP types provided
    M_samp = None
    alpha = 0
    if cp_types1 is not None and cp_types2 is not None:
        # Create cost matrix: 0 for same type, 1 for different type
        M_samp = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                if cp_types1[i] != cp_types2[j]:
                    M_samp[i, j] = 1.0
        alpha = alpha_type
        if verbose:
            print(f"  Type penalty: alpha={alpha_type}")

    # Run POT's COOT
    # X rows = samples (nodes), X cols = features (hyperedges)
    # Returns: pi_samp (node coupling), pi_feat (hyperedge coupling)
    pi, xi, log = coot.co_optimal_transport(
        X=omega1_norm,
        Y=omega2_norm,
        wx_samp=mu1_norm,
        wy_samp=mu2_norm,
        wx_feat=nu1_norm,
        wy_feat=nu2_norm,
        epsilon=epsilon,
        alpha=alpha,
        M_samp=M_samp,
        nits_bcd=nits_bcd,
        tol_bcd=tol_bcd,
        nits_ot=nits_ot,
        tol_sinkhorn=tol_sinkhorn,
        log=True,
        verbose=verbose
    )

    if verbose:
        print(f"  COOT completed.")
        print(f"  π shape: {pi.shape}, ξ shape: {xi.shape}")

    return pi, xi, log


def compute_coot_cost(omega1, omega2, pi, xi):
    """Compute the COOT cost: Σ |ω₁(v,e) - ω₂(v',e')|² π(v,v') ξ(e,e')"""
    # Normalize ω
    scale = max(omega1.max(), omega2.max(), 1e-10)
    omega1_norm = omega1 / scale
    omega2_norm = omega2 / scale

    # Vectorized computation
    # diff[v,v',e,e'] = (omega1[v,e] - omega2[v',e'])^2
    omega1_exp = omega1_norm[:, np.newaxis, :, np.newaxis]
    omega2_exp = omega2_norm[np.newaxis, :, np.newaxis, :]
    diff_sq = (omega1_exp - omega2_exp) ** 2

    # Cost = sum over all (v,v',e,e') of diff_sq * pi[v,v'] * xi[e,e']
    # = sum_{v,v'} pi[v,v'] * sum_{e,e'} diff_sq[v,v',e,e'] * xi[e,e']
    cost = np.einsum('ijkl,ij,kl->', diff_sq, pi, xi)

    return cost


# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================

def analyze_coupling(pi, xi, cp1, cp2, vc1, vc2):
    """Analyze the coupling matrices."""
    results = {}

    # Node coupling analysis
    n1, n2 = pi.shape
    results['n1'] = n1
    results['n2'] = n2

    # Find best matches for each node
    pi_row_max = np.argmax(pi, axis=1)
    pi_col_max = np.argmax(pi, axis=0)

    # Type preservation
    types1 = cp1['cp_type'].values
    types2 = cp2['cp_type'].values

    type_preserved = 0
    for v in range(n1):
        vp = pi_row_max[v]
        if types1[v] == types2[vp]:
            type_preserved += 1
    results['type_preservation'] = type_preserved / n1

    # Hyperedge coupling analysis
    m1, m2 = xi.shape
    results['m1'] = m1
    results['m2'] = m2

    # Coupling entropy (lower = more concentrated)
    pi_entropy = -np.sum(pi * np.log(pi + 1e-10))
    xi_entropy = -np.sum(xi * np.log(xi + 1e-10))
    results['pi_entropy'] = pi_entropy
    results['xi_entropy'] = xi_entropy

    # Maximum coupling values
    results['pi_max'] = pi.max()
    results['xi_max'] = xi.max()

    return results


def plot_hypercot_results(pi, xi, cp1, cp2, vc1, vc2, log, cost, save_path):
    """Visualize HyperCOT results."""
    fig = plt.figure(figsize=(16, 10))

    # 1. Node coupling matrix π
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(pi, aspect='auto', cmap='Blues')
    ax1.set_xlabel('Noisy CP')
    ax1.set_ylabel('Clean CP')
    ax1.set_title('π (Node Coupling)')
    plt.colorbar(im1, ax=ax1)

    # 2. Hyperedge coupling matrix ξ
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(xi, aspect='auto', cmap='Oranges')
    ax2.set_xlabel('Noisy Region')
    ax2.set_ylabel('Clean Region')
    ax2.set_title('ξ (Hyperedge Coupling)')
    plt.colorbar(im2, ax=ax2)

    # 3. Convergence plot (use distances from POT log if available)
    ax3 = fig.add_subplot(2, 3, 3)
    if 'distances' in log:
        ax3.plot(log['distances'], 'b-', linewidth=2, label='Distance')
    else:
        # Fallback: just show final cost
        ax3.bar([0], [cost], color='blue', alpha=0.7)
        ax3.set_xticks([0])
        ax3.set_xticklabels(['Final'])
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Cost/Distance')
    ax3.set_title('Optimization (POT COOT)')
    ax3.legend()
    ax3.grid(True, linestyle=':')

    # 4. Node correspondence visualization
    ax4 = fig.add_subplot(2, 3, 4)
    coords1 = cp1[['x', 'y']].values
    coords2 = cp2[['x', 'y']].values
    types1 = cp1['cp_type'].values
    types2 = cp2['cp_type'].values

    CP_COLORS = {0: '#2166ac', 1: '#4daf4a', 2: '#e41a1c'}

    # Plot clean CPs on left, noisy on right (shifted)
    shift = 120

    for t in [0, 1, 2]:
        mask1 = types1 == t
        mask2 = types2 == t
        ax4.scatter(coords1[mask1, 0], coords1[mask1, 1],
                   c=CP_COLORS[t], s=40, alpha=0.7, marker='o')
        ax4.scatter(coords2[mask2, 0] + shift, coords2[mask2, 1],
                   c=CP_COLORS[t], s=40, alpha=0.7, marker='s')

    # Draw top correspondences
    n_top = 20
    flat_indices = np.argsort(pi.flatten())[-n_top:]
    for idx in flat_indices:
        v, vp = np.unravel_index(idx, pi.shape)
        weight = pi[v, vp]
        ax4.plot([coords1[v, 0], coords2[vp, 0] + shift],
                [coords1[v, 1], coords2[vp, 1]],
                'gray', alpha=min(weight * 50, 0.8), linewidth=1)

    ax4.axvline(x=110, color='black', linestyle='--', alpha=0.3)
    ax4.text(50, 105, 'Clean', ha='center', fontsize=10)
    ax4.text(50 + shift, 105, 'Noisy', ha='center', fontsize=10)
    ax4.set_xlim(-10, 250)
    ax4.set_ylim(-10, 115)
    ax4.set_title(f'Top {n_top} Node Correspondences')
    ax4.set_aspect('equal')

    # 5. π row sums (should equal μ₁)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.bar(range(len(pi)), pi.sum(axis=1), alpha=0.7, label='π row sum')
    ax5.set_xlabel('Clean CP')
    ax5.set_ylabel('Sum')
    ax5.set_title('π Row Marginals (should = μ₁)')
    ax5.legend()

    # 6. ξ row sums (should equal ν₁)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.bar(range(len(xi)), xi.sum(axis=1), alpha=0.7, color='orange', label='ξ row sum')
    ax6.set_xlabel('Clean Region')
    ax6.set_ylabel('Sum')
    ax6.set_title('ξ Row Marginals (should = ν₁)')
    ax6.legend()

    plt.suptitle(f'HyperCOT Results: Clean ↔ Noisy MS Complex\nFinal Cost = {cost:.6f}',
                fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("HyperCOT: CO-OPTIMAL TRANSPORT FOR MS COMPLEX HYPERGRAPHS")
    print("="*60)
    print("""
Objective:
    min_{π, ξ} Σ |ω₁(v,e) - ω₂(v',e')|² π(v,v') ξ(e,e')

Subject to:
    π marginals = (μ₁, μ₂)
    ξ marginals = (ν₁, ν₂)
""")

    # Load data
    print("\n1. Loading data...")
    clean = load_hypercot_data("clean")
    noisy = load_hypercot_data("noisy")

    print(f"   Clean: {len(clean['mu'])} CPs, {len(clean['nu'])} regions")
    print(f"   Noisy: {len(noisy['mu'])} CPs, {len(noisy['nu'])} regions")
    print(f"   Clean ω shape: {clean['omega'].shape}")
    print(f"   Noisy ω shape: {noisy['omega'].shape}")

    # Run COOT using POT library with type penalty
    print("\n2. Running POT COOT optimization...")
    pi, xi, log = run_coot(
        clean['omega'], noisy['omega'],
        clean['mu'], noisy['mu'],
        clean['nu'], noisy['nu'],
        cp_types1=clean['cp_data']['cp_type'].values,
        cp_types2=noisy['cp_data']['cp_type'].values,
        alpha_type=0.001,  # Very small type penalty for ~95%
        epsilon=0.001,    # Sinkhorn regularization
        nits_bcd=100,    # BCD iterations
        tol_bcd=1e-7,    # BCD tolerance
        nits_ot=500,     # OT iterations per BCD step
        tol_sinkhorn=1e-7,
        verbose=True
    )

    # Compute final cost
    cost = compute_coot_cost(clean['omega'], noisy['omega'], pi, xi)

    print(f"\n   Final COOT cost: {cost:.6f}")
    print(f"   π shape: {pi.shape}")
    print(f"   ξ shape: {xi.shape}")

    # Analyze results
    print("\n3. Analyzing coupling...")
    analysis = analyze_coupling(pi, xi, clean['cp_data'], noisy['cp_data'],
                                clean['vc_data'], noisy['vc_data'])

    print(f"   Type preservation: {analysis['type_preservation']*100:.1f}%")
    print(f"   π entropy: {analysis['pi_entropy']:.4f}")
    print(f"   ξ entropy: {analysis['xi_entropy']:.4f}")
    print(f"   π max coupling: {analysis['pi_max']:.6f}")
    print(f"   ξ max coupling: {analysis['xi_max']:.6f}")

    # Verify marginals
    print("\n   Marginal verification:")
    print(f"   π row sum range: [{pi.sum(axis=1).min():.6f}, {pi.sum(axis=1).max():.6f}]")
    print(f"   π col sum range: [{pi.sum(axis=0).min():.6f}, {pi.sum(axis=0).max():.6f}]")
    print(f"   ξ row sum range: [{xi.sum(axis=1).min():.6f}, {xi.sum(axis=1).max():.6f}]")
    print(f"   ξ col sum range: [{xi.sum(axis=0).min():.6f}, {xi.sum(axis=0).max():.6f}]")

    # Visualize
    print("\n4. Generating visualization...")
    plot_path = os.path.join(BASE_PATH, "hypercot_results.png")
    plot_hypercot_results(pi, xi, clean['cp_data'], noisy['cp_data'],
                         clean['vc_data'], noisy['vc_data'], log, cost, plot_path)

    # Save results
    print("\n5. Saving results...")

    # Save π
    pi_df = pd.DataFrame(pi,
                        index=[f'Clean_CP_{i}' for i in range(pi.shape[0])],
                        columns=[f'Noisy_CP_{i}' for i in range(pi.shape[1])])
    pi_path = os.path.join(BASE_PATH, "hypercot_pi.csv")
    pi_df.to_csv(pi_path)
    print(f"   Saved: {pi_path}")

    # Save ξ
    xi_df = pd.DataFrame(xi,
                        index=[f'Clean_Region_{i+1}' for i in range(xi.shape[0])],
                        columns=[f'Noisy_Region_{i+1}' for i in range(xi.shape[1])])
    xi_path = os.path.join(BASE_PATH, "hypercot_xi.csv")
    xi_df.to_csv(xi_path)
    print(f"   Saved: {xi_path}")

    # Save summary
    summary = {
        'coot_cost': cost,
        'n_clean_cps': len(clean['mu']),
        'n_noisy_cps': len(noisy['mu']),
        'n_clean_regions': len(clean['nu']),
        'n_noisy_regions': len(noisy['nu']),
        'type_preservation': analysis['type_preservation'],
        'pi_entropy': analysis['pi_entropy'],
        'xi_entropy': analysis['xi_entropy'],
        'pi_max': analysis['pi_max'],
        'xi_max': analysis['xi_max']
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(BASE_PATH, "hypercot_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"   Saved: {summary_path}")

    # Print final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"""
COOT Distance: {cost:.6f}

Coupling Matrices:
  π (nodes):      {pi.shape[0]} x {pi.shape[1]}  (max: {analysis['pi_max']:.6f})
  ξ (hyperedges): {xi.shape[0]} x {xi.shape[1]}  (max: {analysis['xi_max']:.6f})

Quality Metrics:
  Type preservation: {analysis['type_preservation']*100:.1f}%
  π entropy: {analysis['pi_entropy']:.4f}
  ξ entropy: {analysis['xi_entropy']:.4f}

Output Files:
  - hypercot_pi.csv (node coupling)
  - hypercot_xi.csv (hyperedge coupling)
  - hypercot_summary.csv (metrics)
  - hypercot_results.png (visualization)
""")

    print("="*60)
    print("DONE!")
    print("="*60)

    return {
        'pi': pi,
        'xi': xi,
        'cost': cost,
        'log': log,
        'analysis': analysis
    }


if __name__ == "__main__":
    results = main()
