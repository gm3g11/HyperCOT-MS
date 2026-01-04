"""
Compute ν (Hyperedge Measure) for MS Complex

ν(e) = sum of boundary CP measures
     = Σ_{v ∈ boundary(e)} μ(v)

Then normalized so that Σ_e ν(e) = 1

Each hyperedge has 4 boundary CPs: 1 min + 2 saddles + 1 max
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_PATH = "/Users/gmeng/Desktop/COOT on Morse-Smale"


def load_data(prefix):
    """Load μ (node measure) and hypergraph data."""
    # Load μ
    mu_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_node_measure.csv"))

    # Load hypergraph
    hyper_df = pd.read_csv(os.path.join(BASE_PATH, f"hypergraph_{prefix}.csv"))

    # Load virtual centers (for additional info)
    vc_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_virtual_centers.csv"))

    return mu_df, hyper_df, vc_df


def compute_nu(mu_df, hyper_df):
    """
    Compute ν (hyperedge measure) as sum of boundary CP measures.

    ν(e) = Σ_{v ∈ boundary(e)} μ(v)

    Returns:
        nu: Array of hyperedge measures (normalized)
        nu_unnorm: Unnormalized measures
        nu_details: DataFrame with detailed breakdown
    """
    n_hyper = len(hyper_df)
    mu_values = mu_df['mu'].values

    nu_unnorm = np.zeros(n_hyper)
    nu_details = []

    for idx, row in hyper_df.iterrows():
        min_id = int(row['min_id'])
        max_id = int(row['max_id'])
        saddle_ids = eval(row['boundary_saddles'])

        # Boundary CPs
        boundary_cps = [min_id] + list(saddle_ids) + [max_id]

        # Sum of μ values
        mu_sum = sum(mu_values[cp] for cp in boundary_cps)
        nu_unnorm[idx] = mu_sum

        # Store details
        nu_details.append({
            'region_id': idx + 1,
            'min_id': min_id,
            'saddle1_id': saddle_ids[0],
            'saddle2_id': saddle_ids[1],
            'max_id': max_id,
            'mu_min': mu_values[min_id],
            'mu_saddle1': mu_values[saddle_ids[0]],
            'mu_saddle2': mu_values[saddle_ids[1]],
            'mu_max': mu_values[max_id],
            'nu_unnorm': mu_sum
        })

    # Normalize
    total = nu_unnorm.sum()
    nu = nu_unnorm / total

    # Add normalized values to details
    nu_details_df = pd.DataFrame(nu_details)
    nu_details_df['nu'] = nu

    return nu, nu_unnorm, nu_details_df


def plot_nu_results(prefix, nu, nu_details_df, hyper_df, save_path):
    """Visualize ν distribution and comparison with μ."""
    n_hyper = len(nu)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. ν bar chart
    ax1 = axes[0]
    colors = plt.cm.viridis(nu / nu.max())
    bars = ax1.bar(range(1, n_hyper + 1), nu, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(1/n_hyper, color='red', linestyle='--', label=f'Uniform: {1/n_hyper:.4f}')
    ax1.set_xlabel('Region (Hyperedge)')
    ax1.set_ylabel('ν (Probability)')
    ax1.set_title(f'{prefix}: ν (Hyperedge Measure)')
    ax1.legend()
    ax1.set_xlim(0, n_hyper + 1)

    # 2. ν distribution histogram
    ax2 = axes[1]
    ax2.hist(nu, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(nu.mean(), color='red', linestyle='--', label=f'Mean: {nu.mean():.4f}')
    ax2.axvline(np.median(nu), color='orange', linestyle='--', label=f'Median: {np.median(nu):.4f}')
    ax2.axvline(1/n_hyper, color='green', linestyle=':', label=f'Uniform: {1/n_hyper:.4f}')
    ax2.set_xlabel('ν')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{prefix}: ν Distribution')
    ax2.legend()

    # 3. Component breakdown (stacked bar showing μ contributions)
    ax3 = axes[2]

    x = np.arange(n_hyper)
    width = 0.8

    # Stack the 4 μ components
    mu_min = nu_details_df['mu_min'].values
    mu_s1 = nu_details_df['mu_saddle1'].values
    mu_s2 = nu_details_df['mu_saddle2'].values
    mu_max = nu_details_df['mu_max'].values

    # Normalize each to show proportion within each region
    totals = mu_min + mu_s1 + mu_s2 + mu_max

    ax3.bar(x + 1, mu_min / totals, width, label='μ(min)', color='#2166ac')
    ax3.bar(x + 1, mu_s1 / totals, width, bottom=mu_min/totals, label='μ(saddle1)', color='#4daf4a')
    ax3.bar(x + 1, mu_s2 / totals, width, bottom=(mu_min+mu_s1)/totals, label='μ(saddle2)', color='#66c2a5')
    ax3.bar(x + 1, mu_max / totals, width, bottom=(mu_min+mu_s1+mu_s2)/totals, label='μ(max)', color='#e41a1c')

    ax3.set_xlabel('Region (Hyperedge)')
    ax3.set_ylabel('Proportion')
    ax3.set_title(f'{prefix}: μ Contribution Breakdown')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xlim(0, n_hyper + 1)
    ax3.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {save_path}")


def process_ms_complex(prefix):
    """Process a single MS complex: compute ν."""
    print(f"\n{'='*60}")
    print(f"Processing {prefix} MS complex")
    print(f"{'='*60}")

    # Load data
    print("\n1. Loading data...")
    mu_df, hyper_df, vc_df = load_data(prefix)
    n_cp = len(mu_df)
    n_hyper = len(hyper_df)
    print(f"   CPs: {n_cp}")
    print(f"   Hyperedges (regions): {n_hyper}")
    print(f"   μ sum: {mu_df['mu'].sum():.6f}")

    # Compute ν
    print("\n2. Computing ν...")
    nu, nu_unnorm, nu_details_df = compute_nu(mu_df, hyper_df)

    print(f"   ν shape: {nu.shape}")
    print(f"   ν sum: {nu.sum():.6f}")
    print(f"   ν range: [{nu.min():.6f}, {nu.max():.6f}]")
    print(f"   ν mean: {nu.mean():.6f}")
    print(f"   ν median: {np.median(nu):.6f}")
    print(f"   Uniform would be: {1/n_hyper:.6f}")

    # Entropy comparison
    nu_entropy = -np.sum(nu * np.log(nu + 1e-10))
    uniform_entropy = np.log(n_hyper)
    print(f"\n   ν entropy: {nu_entropy:.4f}")
    print(f"   Uniform entropy: {uniform_entropy:.4f}")
    print(f"   Entropy ratio: {nu_entropy/uniform_entropy:.4f}")

    # Generate visualization
    print("\n3. Generating visualization...")
    plot_path = os.path.join(BASE_PATH, f"{prefix}_nu.png")
    plot_nu_results(prefix, nu, nu_details_df, hyper_df, plot_path)

    # Save results
    print("\n4. Saving results...")

    # Save ν vector
    nu_df = pd.DataFrame({
        'region_id': range(1, n_hyper + 1),
        'nu': nu,
        'nu_unnorm': nu_unnorm
    })
    nu_csv_path = os.path.join(BASE_PATH, f"{prefix}_nu.csv")
    nu_df.to_csv(nu_csv_path, index=False)
    print(f"   Saved: {nu_csv_path}")

    # Save detailed breakdown
    details_csv_path = os.path.join(BASE_PATH, f"{prefix}_nu_details.csv")
    nu_details_df.to_csv(details_csv_path, index=False)
    print(f"   Saved: {details_csv_path}")

    return {
        'nu': nu,
        'nu_unnorm': nu_unnorm,
        'nu_details': nu_details_df
    }


def main():
    print("="*60)
    print("COMPUTING ν (HYPEREDGE MEASURE) FOR MS COMPLEXES")
    print("="*60)
    print("""
Formula: ν(e) = Σ_{v ∈ boundary(e)} μ(v)

Each hyperedge has 4 boundary CPs:
  - 1 minimum
  - 2 saddles
  - 1 maximum

Then normalized: ν(e) = ν(e) / Σ_e ν(e)
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
    print(f"  ν shape: {clean_results['nu'].shape}")
    print(f"  ν mean: {clean_results['nu'].mean():.6f}")
    print(f"  ν std: {clean_results['nu'].std():.6f}")

    print("\nNoisy MS Complex:")
    print(f"  ν shape: {noisy_results['nu'].shape}")
    print(f"  ν mean: {noisy_results['nu'].mean():.6f}")
    print(f"  ν std: {noisy_results['nu'].std():.6f}")

    print("\nOutput files:")
    print("  - clean_nu.csv, noisy_nu.csv (ν vectors)")
    print("  - clean_nu_details.csv, noisy_nu_details.csv (breakdown)")
    print("  - clean_nu.png, noisy_nu.png (visualizations)")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

    return clean_results, noisy_results


if __name__ == "__main__":
    clean_results, noisy_results = main()
