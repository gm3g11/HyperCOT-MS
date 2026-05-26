#!/usr/bin/env python3
"""Compute WD/GWD/FGW/MS-COOT for consecutive timestep pairs."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mscoot.config import get_dataset_dir, get_results_dir
from mscoot.data_adapter import DatasetAdapter
from mscoot.graph.adjacency import build_adjacency_from_edges, compute_graph_distances
from mscoot.solvers.wasserstein import compute_wd
from mscoot.solvers.gromov import compute_gwd
from mscoot.solvers.fused_gw import compute_fgw, build_feature_cost
from mscoot.solvers.mscoot_solver import compute_mscoot
from mscoot.utils import (compute_type_preservation, compute_spatial_coherence,
                            compute_coupling_entropy)


def load_coot_data(coot_dir, ts):
    """Load precomputed mu, nu, omega for a timestep."""
    mu_df = pd.read_csv(coot_dir / f"mu_{ts:03d}.csv")
    nu_df = pd.read_csv(coot_dir / f"nu_{ts:03d}.csv")
    omega_df = pd.read_csv(coot_dir / f"omega_{ts:03d}.csv", index_col=0)

    return {
        'mu': mu_df['mu'].values,
        'nu': nu_df['nu'].values,
        'omega': omega_df.values,
        'cp_types': mu_df['cp_type'].values,
        'coords': mu_df[['x', 'y', 'z']].values if 'z' in mu_df.columns else mu_df[['x', 'y']].values,
        'scalars': mu_df['scalar'].values,
        'n_cps': len(mu_df),
        'n_regions': len(nu_df),
    }


def main():
    parser = argparse.ArgumentParser(description='Compute consecutive pair metrics')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--methods', type=str, default='wd,gwd,fgw,mscoot',
                        help='Comma-separated methods')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.001)
    parser.add_argument('--spatial-weight', type=float, default=1.0)
    args = parser.parse_args()

    dataset_dir = get_dataset_dir(args.dataset)
    results_dir = get_results_dir(args.dataset)
    coot_dir = results_dir / "coot_data"
    couplings_dir = results_dir / "couplings"
    metrics_dir = results_dir / "metrics"
    couplings_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    adapter = DatasetAdapter(dataset_dir)
    diagonal = adapter.get_domain_diagonal()
    methods = [m.strip().lower() for m in args.methods.split(',')]

    # Find valid consecutive pairs
    stats = pd.read_csv(coot_dir / "component_stats.csv")
    valid_ts = sorted(stats[stats['status'] == 'ok']['timestep'].values)
    pairs = [(valid_ts[i], valid_ts[i+1])
             for i in range(len(valid_ts) - 1)
             if valid_ts[i+1] == valid_ts[i] + 1]

    print(f"Processing {len(pairs)} consecutive pairs...")

    # Load global data for GWD/FGW
    need_graph = 'gwd' in methods or 'fgw' in methods

    all_metrics = []
    for ts1, ts2 in tqdm(pairs, desc="Processing pairs"):
        data1 = load_coot_data(coot_dir, ts1)
        data2 = load_coot_data(coot_dir, ts2)

        row = {'ts1': ts1, 'ts2': ts2,
               'n_cps_1': data1['n_cps'], 'n_cps_2': data2['n_cps'],
               'n_regions_1': data1['n_regions'], 'n_regions_2': data2['n_regions']}

        # WD
        if 'wd' in methods:
            coupling, dist = compute_wd(data1['coords'], data2['coords'], diagonal)
            row['wd_distance'] = dist
            row['wd_type_preservation'] = compute_type_preservation(
                coupling, data1['cp_types'], data2['cp_types'])
            row['wd_spatial_coherence'] = compute_spatial_coherence(
                coupling, data1['coords'], data2['coords'])
            row['wd_entropy'] = compute_coupling_entropy(coupling)

        # GWD
        if 'gwd' in methods:
            edges1 = adapter.load_edges(ts1)
            edges2 = adapter.load_edges(ts2)
            adj1 = build_adjacency_from_edges(data1['n_cps'], edges1)
            adj2 = build_adjacency_from_edges(data2['n_cps'], edges2)
            D1 = compute_graph_distances(adj1, data1['coords'])
            D2 = compute_graph_distances(adj2, data2['coords'])

            coupling, dist = compute_gwd(D1, D2)
            row['gwd_distance'] = dist
            row['gwd_type_preservation'] = compute_type_preservation(
                coupling, data1['cp_types'], data2['cp_types'])
            row['gwd_spatial_coherence'] = compute_spatial_coherence(
                coupling, data1['coords'], data2['coords'])
            row['gwd_entropy'] = compute_coupling_entropy(coupling)

        # FGW
        if 'fgw' in methods:
            if 'gwd' not in methods:
                edges1 = adapter.load_edges(ts1)
                edges2 = adapter.load_edges(ts2)
                adj1 = build_adjacency_from_edges(data1['n_cps'], edges1)
                adj2 = build_adjacency_from_edges(data2['n_cps'], edges2)
                D1 = compute_graph_distances(adj1, data1['coords'])
                D2 = compute_graph_distances(adj2, data2['coords'])

            M = build_feature_cost(data1['coords'], data2['coords'], diagonal)
            coupling, dist = compute_fgw(M, D1, D2, alpha=args.alpha)
            row['fgw_distance'] = dist
            row['fgw_type_preservation'] = compute_type_preservation(
                coupling, data1['cp_types'], data2['cp_types'])
            row['fgw_spatial_coherence'] = compute_spatial_coherence(
                coupling, data1['coords'], data2['coords'])
            row['fgw_entropy'] = compute_coupling_entropy(coupling)

        # MS-COOT
        if 'mscoot' in methods:
            # mu/nu are already floored from compute_components; use directly
            mu1 = data1['mu']
            mu2 = data2['mu']
            nu1 = data1['nu']
            nu2 = data2['nu']

            pi, xi, cost = compute_mscoot(
                data1['omega'], data2['omega'],
                mu1, mu2, nu1, nu2,
                types1=data1['cp_types'], types2=data2['cp_types'],
                alpha=args.alpha, epsilon=args.epsilon,
                coords1=data1['coords'], coords2=data2['coords'],
                spatial_weight=args.spatial_weight,
            )

            row['coot_distance'] = cost
            row['coot_type_preservation'] = compute_type_preservation(
                pi, data1['cp_types'], data2['cp_types'])
            row['coot_spatial_coherence'] = compute_spatial_coherence(
                pi, data1['coords'], data2['coords'])
            row['coot_pi_entropy'] = compute_coupling_entropy(pi)
            row['coot_xi_entropy'] = compute_coupling_entropy(xi)

            # Save couplings
            pd.DataFrame(pi).to_csv(
                couplings_dir / f"pi_{ts1:03d}_{ts2:03d}.csv", index=False)
            pd.DataFrame(xi).to_csv(
                couplings_dir / f"xi_{ts1:03d}_{ts2:03d}.csv", index=False)

        all_metrics.append(row)

    # Save metrics (merge with existing if running subset of methods)
    df = pd.DataFrame(all_metrics)
    out_path = metrics_dir / "consecutive_metrics.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        # Merge: new data takes precedence for shared columns
        merged = existing.merge(df, on=['ts1', 'ts2'], how='outer',
                                suffixes=('_old', ''))
        # Drop _old duplicates, keeping new values
        for col in list(merged.columns):
            if col.endswith('_old'):
                base = col[:-4]
                if base in merged.columns:
                    merged[base] = merged[base].fillna(merged[col])
                else:
                    merged.rename(columns={col: base}, inplace=True)
                    continue
                merged.drop(columns=[col], inplace=True)
        df = merged
    df.to_csv(out_path, index=False)
    print(f"\nSaved metrics to {out_path}")

    # Print summary
    print("\n=== Summary ===")
    for method in methods:
        prefix = 'coot' if method == 'mscoot' else method
        tp_col = f'{prefix}_type_preservation'
        sc_col = f'{prefix}_spatial_coherence'
        if tp_col in df.columns:
            print(f"  {method.upper()}: type_pres={df[tp_col].mean():.1%} +/- {df[tp_col].std():.1%}, "
                  f"spatial_coh={df[sc_col].mean():.1f}")


if __name__ == '__main__':
    main()
