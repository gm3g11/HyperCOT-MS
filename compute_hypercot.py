"""
HyperCOT: Co-Optimal Transport for Hypergraphs

Computes the optimal coupling between two MS complex hypergraphs.

Objective:
    min_{π, ξ} Σ_{v,v'} Σ_{e,e'} |ω₁(v,e) - ω₂(v',e')|² π(v,v') ξ(e,e')

Subject to:
    - π has marginals μ₁ and μ₂ (node measures)
    - ξ has marginals ν₁ and ν₂ (hyperedge measures)

Algorithm: Alternating optimization with Sinkhorn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
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
# SINKHORN ALGORITHM
# =============================================================================

def sinkhorn(C, a, b, reg=0.01, max_iter=1000, tol=1e-9):
    """
    Sinkhorn algorithm for entropy-regularized optimal transport.

    Args:
        C: Cost matrix (n x m)
        a: Source marginal (n,)
        b: Target marginal (m,)
        reg: Regularization parameter
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        P: Transport plan (n x m)
    """
    n, m = C.shape
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Normalize marginals
    a = a / a.sum()
    b = b / b.sum()

    # Initialize
    K = np.exp(-C / reg)
    K = np.clip(K, 1e-300, None)  # Avoid numerical issues

    u = np.ones(n)
    v = np.ones(m)

    for i in range(max_iter):
        u_prev = u.copy()

        # Update u and v
        v = b / (K.T @ u + 1e-300)
        u = a / (K @ v + 1e-300)

        # Check convergence
        if np.max(np.abs(u - u_prev)) < tol:
            break

    # Compute transport plan
    P = np.diag(u) @ K @ np.diag(v)

    return P


# =============================================================================
# HYPERCOT ALGORITHM
# =============================================================================

def compute_cost_tensor(omega1, omega2):
    """
    Compute the 4D cost tensor L[v,v',e,e'] = |ω₁(v,e) - ω₂(v',e')|².

    For efficiency, we don't explicitly form the 4D tensor.
    Instead, we compute costs as needed during optimization.
    """
    # Normalize ω matrices to [0, 1] for numerical stability
    omega1_norm = omega1 / (omega1.max() + 1e-10)
    omega2_norm = omega2 / (omega2.max() + 1e-10)

    return omega1_norm, omega2_norm


def compute_node_cost(omega1, omega2, xi):
    """
    Compute cost matrix for node coupling given hyperedge coupling ξ.

    C_π[v,v'] = Σ_{e,e'} |ω₁(v,e) - ω₂(v',e')|² ξ(e,e')
    """
    n1, m1 = omega1.shape  # n1 CPs, m1 hyperedges in graph 1
    n2, m2 = omega2.shape  # n2 CPs, m2 hyperedges in graph 2

    C_pi = np.zeros((n1, n2))

    for v in range(n1):
        for vp in range(n2):
            # Compute weighted sum over hyperedge pairs
            cost = 0.0
            for e in range(m1):
                for ep in range(m2):
                    diff = omega1[v, e] - omega2[vp, ep]
                    cost += (diff ** 2) * xi[e, ep]
            C_pi[v, vp] = cost

    return C_pi


def compute_node_cost_vectorized(omega1, omega2, xi):
    """Vectorized version of compute_node_cost."""
    n1, m1 = omega1.shape
    n2, m2 = omega2.shape

    # Reshape for broadcasting
    # omega1: (n1, m1) -> (n1, 1, m1, 1)
    # omega2: (n2, m2) -> (1, n2, 1, m2)
    # diff: (n1, n2, m1, m2)
    omega1_exp = omega1[:, np.newaxis, :, np.newaxis]
    omega2_exp = omega2[np.newaxis, :, np.newaxis, :]

    diff_sq = (omega1_exp - omega2_exp) ** 2  # (n1, n2, m1, m2)

    # Weight by xi and sum over hyperedge pairs
    # xi: (m1, m2)
    C_pi = np.einsum('ijkl,kl->ij', diff_sq, xi)

    return C_pi


def compute_hyperedge_cost_vectorized(omega1, omega2, pi):
    """
    Compute cost matrix for hyperedge coupling given node coupling π.

    C_ξ[e,e'] = Σ_{v,v'} |ω₁(v,e) - ω₂(v',e')|² π(v,v')
    """
    n1, m1 = omega1.shape
    n2, m2 = omega2.shape

    # Reshape for broadcasting
    omega1_exp = omega1[:, np.newaxis, :, np.newaxis]
    omega2_exp = omega2[np.newaxis, :, np.newaxis, :]

    diff_sq = (omega1_exp - omega2_exp) ** 2  # (n1, n2, m1, m2)

    # Weight by pi and sum over node pairs
    # pi: (n1, n2)
    C_xi = np.einsum('ijkl,ij->kl', diff_sq, pi)

    return C_xi


def hypercot(omega1, omega2, mu1, mu2, nu1, nu2,
             reg_pi=0.01, reg_xi=0.01, max_iter=100, tol=1e-6, verbose=True):
    """
    HyperCOT: Co-Optimal Transport for Hypergraphs.

    Alternating optimization between π (node coupling) and ξ (hyperedge coupling).

    Args:
        omega1, omega2: Hypernetwork functions (n1 x m1) and (n2 x m2)
        mu1, mu2: Node measures
        nu1, nu2: Hyperedge measures
        reg_pi, reg_xi: Regularization for Sinkhorn
        max_iter: Maximum alternating iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        pi: Node coupling (n1 x n2)
        xi: Hyperedge coupling (m1 x m2)
        cost: Final HyperCOT cost
        history: Optimization history
    """
    n1, m1 = omega1.shape
    n2, m2 = omega2.shape

    # Normalize ω for numerical stability
    omega1_norm, omega2_norm = compute_cost_tensor(omega1, omega2)

    # Initialize couplings as outer product of marginals
    pi = np.outer(mu1, mu2)
    xi = np.outer(nu1, nu2)

    history = {'cost': [], 'pi_change': [], 'xi_change': []}

    if verbose:
        print(f"  Starting HyperCOT optimization...")
        print(f"  Nodes: {n1} x {n2}, Hyperedges: {m1} x {m2}")

    for iteration in range(max_iter):
        pi_old = pi.copy()
        xi_old = xi.copy()

        # Step 1: Fix ξ, optimize π
        C_pi = compute_node_cost_vectorized(omega1_norm, omega2_norm, xi)
        pi = sinkhorn(C_pi, mu1, mu2, reg=reg_pi)

        # Step 2: Fix π, optimize ξ
        C_xi = compute_hyperedge_cost_vectorized(omega1_norm, omega2_norm, pi)
        xi = sinkhorn(C_xi, nu1, nu2, reg=reg_xi)

        # Compute cost
        cost = np.sum(C_pi * pi)

        # Check convergence
        pi_change = np.max(np.abs(pi - pi_old))
        xi_change = np.max(np.abs(xi - xi_old))

        history['cost'].append(cost)
        history['pi_change'].append(pi_change)
        history['xi_change'].append(xi_change)

        if verbose and (iteration + 1) % 10 == 0:
            print(f"    Iter {iteration+1}: cost={cost:.6f}, "
                  f"Δπ={pi_change:.2e}, Δξ={xi_change:.2e}")

        if pi_change < tol and xi_change < tol:
            if verbose:
                print(f"  Converged at iteration {iteration+1}")
            break

    # Final cost computation (unregularized)
    final_cost = compute_hypercot_cost(omega1, omega2, pi, xi)

    return pi, xi, final_cost, history


def compute_hypercot_cost(omega1, omega2, pi, xi):
    """Compute the actual HyperCOT cost (without regularization)."""
    n1, m1 = omega1.shape
    n2, m2 = omega2.shape

    # Normalize ω
    scale = max(omega1.max(), omega2.max())
    omega1_norm = omega1 / scale
    omega2_norm = omega2 / scale

    # Compute full cost
    cost = 0.0
    for v in range(n1):
        for vp in range(n2):
            for e in range(m1):
                for ep in range(m2):
                    diff = omega1_norm[v, e] - omega2_norm[vp, ep]
                    cost += (diff ** 2) * pi[v, vp] * xi[e, ep]

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


def plot_hypercot_results(pi, xi, cp1, cp2, vc1, vc2, history, cost, save_path):
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

    # 3. Convergence plot
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(history['cost'], 'b-', linewidth=2, label='Cost')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Cost')
    ax3.set_title('Optimization Convergence')
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

    # Run HyperCOT
    print("\n2. Running HyperCOT optimization...")
    pi, xi, cost, history = hypercot(
        clean['omega'], noisy['omega'],
        clean['mu'], noisy['mu'],
        clean['nu'], noisy['nu'],
        reg_pi=0.05,  # Regularization for nodes
        reg_xi=0.05,  # Regularization for hyperedges
        max_iter=100,
        tol=1e-6,
        verbose=True
    )

    print(f"\n   Final HyperCOT cost: {cost:.6f}")
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
                         clean['vc_data'], noisy['vc_data'], history, cost, plot_path)

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
        'hypercot_cost': cost,
        'n_clean_cps': len(clean['mu']),
        'n_noisy_cps': len(noisy['mu']),
        'n_clean_regions': len(clean['nu']),
        'n_noisy_regions': len(noisy['nu']),
        'type_preservation': analysis['type_preservation'],
        'pi_entropy': analysis['pi_entropy'],
        'xi_entropy': analysis['xi_entropy'],
        'iterations': len(history['cost'])
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
HyperCOT Distance: {cost:.6f}

Coupling Matrices:
  π (nodes):      {pi.shape[0]} x {pi.shape[1]}
  ξ (hyperedges): {xi.shape[0]} x {xi.shape[1]}

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
        'history': history,
        'analysis': analysis
    }


if __name__ == "__main__":
    results = main()
