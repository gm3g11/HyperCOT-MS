#!/usr/bin/env python3
"""Compute MS-COOT components (hypergraphs, mu, nu, omega) for a dataset."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mscoot.config import load_dataset_config, get_dataset_dir, get_results_dir
from mscoot.data_adapter import DatasetAdapter
from mscoot.graph.hypergraph import build_hypergraph
from mscoot.measures.node_measure import compute_mu
from mscoot.measures.edge_measure import compute_nu
from mscoot.measures.omega import compute_omega


def compute_hypergraphs(adapter, results_dir):
    """Generate hypergraphs for all timesteps."""
    hyper_dir = results_dir / "hypergraphs"
    hyper_dir.mkdir(parents=True, exist_ok=True)

    timesteps = adapter.get_timesteps()
    print(f"Generating hypergraphs for {len(timesteps)} timesteps...")

    # Need raw data for segmentation-based hypergraph construction
    config = adapter.config
    if config.is_temporal:
        cp_raw = pd.read_csv(config.ttk_output_dir / config.files['cp_pattern'])
        seg_raw = pd.read_csv(config.ttk_output_dir / config.files.get('seg_pattern', 'segmentation.csv'))
        sep_raw = pd.read_csv(config.ttk_output_dir / config.files.get('sep_pattern', 'separatrices.csv'))

        # Load persistence diagram if available (used for 3D region filtering)
        pers_path = config.ttk_output_dir / 'persistence_diagram.csv'
        pers_raw = pd.read_csv(pers_path) if pers_path.exists() else None

        ts_col = config.columns.get('timestep', 'TimeStep')

        for ts in timesteps:
            cp_ts = cp_raw[cp_raw[ts_col] == ts]
            seg_ts = seg_raw[seg_raw[ts_col] == ts]
            sep_ts = sep_raw[sep_raw[ts_col] == ts]
            pers_ts = pers_raw[pers_raw[ts_col] == ts] if pers_raw is not None else None

            hg = build_hypergraph(cp_ts, seg_ts, sep_ts,
                                  pers_df=pers_ts, min_persistence=0.0)
            if hg is not None:
                hg.to_csv(hyper_dir / f"hypergraph_{ts:03d}.csv", index=False)
                print(f"  ts={ts:03d}: {len(hg)} regions")
            else:
                print(f"  ts={ts:03d}: SKIPPED")

    print(f"Hypergraphs saved to {hyper_dir}/")


def compute_coot_components(adapter, results_dir, sigma=0.3):
    """Compute mu, nu, omega for all timesteps."""
    coot_dir = results_dir / "coot_data"
    coot_dir.mkdir(parents=True, exist_ok=True)
    hyper_dir = results_dir / "hypergraphs"
    config = adapter.config

    timesteps = adapter.get_timesteps()
    print(f"Computing COOT components (sigma={sigma}) for {len(timesteps)} timesteps...")

    stats = []
    for ts in timesteps:
        cp_data = adapter.load_critical_points(ts)
        edges = adapter.load_edges(ts)

        hyper_path = hyper_dir / f"hypergraph_{ts:03d}.csv"
        if not hyper_path.exists():
            print(f"  ts={ts:03d}: SKIPPED (no hypergraph)")
            stats.append({'timestep': ts, 'status': 'skipped', 'n_cps': len(cp_data)})
            continue

        hyper_df = pd.read_csv(hyper_path)
        if len(hyper_df) == 0:
            stats.append({'timestep': ts, 'status': 'skipped', 'n_cps': len(cp_data)})
            continue

        n_cp = len(cp_data)

        # Compute mu (raw persistence-based weights, no floor)
        mu = compute_mu(cp_data['scalar'].values, edges, sigma=sigma)

        # Compute nu
        nu, members = compute_nu(mu, hyper_df)

        # Compute omega
        coords_3d = np.column_stack([
            cp_data['x'].values, cp_data['y'].values,
            cp_data['z'].values if 'z' in cp_data else np.zeros(n_cp)
        ])
        omega_mat, vcs = compute_omega(
            coords_3d, cp_data['cp_type'].values, edges, members
        )

        # Save
        mu_dict = {
            'cp_index': range(n_cp),
            'cp_type': cp_data['cp_type'].values,
            'x': cp_data['x'].values,
            'y': cp_data['y'].values,
        }
        if config.n_dimensions >= 3:
            mu_dict['z'] = cp_data['z'].values
        mu_dict['scalar'] = cp_data['scalar'].values
        mu_dict['mu'] = mu
        mu_df = pd.DataFrame(mu_dict)
        mu_df.to_csv(coot_dir / f"mu_{ts:03d}.csv", index=False)

        nu_df = pd.DataFrame({'region_idx': range(len(nu)), 'nu': nu})
        nu_df.to_csv(coot_dir / f"nu_{ts:03d}.csv", index=False)

        omega_df = pd.DataFrame(
            omega_mat,
            index=[f'CP_{i}' for i in range(n_cp)],
            columns=[f'Region_{i}' for i in range(omega_mat.shape[1])]
        )
        omega_df.to_csv(coot_dir / f"omega_{ts:03d}.csv")

        vc_df = pd.DataFrame({
            'region_idx': range(len(vcs)),
            'vc_x': vcs[:, 0], 'vc_y': vcs[:, 1], 'vc_z': vcs[:, 2],
            'boundary_cps': [str(m) for m in members],
        })
        vc_df.to_csv(coot_dir / f"vc_{ts:03d}.csv", index=False)

        mu_ent = -np.sum(mu * np.log(mu + 1e-10))
        stats.append({
            'timestep': ts, 'n_cps': n_cp, 'n_regions': len(nu),
            'mu_entropy': round(mu_ent, 4), 'status': 'ok',
        })
        print(f"  ts={ts:03d}: {n_cp:3d} CPs, {len(nu):3d} regions, mu_H={mu_ent:.2f}")

    pd.DataFrame(stats).to_csv(coot_dir / "component_stats.csv", index=False)
    print(f"Components saved to {coot_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Compute MS-COOT components')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--step', choices=['hypergraph', 'components', 'all'],
                        default='all', help='Which step to run')
    parser.add_argument('--sigma', type=float, default=0.3,
                        help='Persistence image kernel width')
    args = parser.parse_args()

    dataset_dir = get_dataset_dir(args.dataset)
    results_dir = get_results_dir(args.dataset)
    results_dir.mkdir(parents=True, exist_ok=True)

    adapter = DatasetAdapter(dataset_dir)

    if args.step in ('hypergraph', 'all'):
        compute_hypergraphs(adapter, results_dir)

    if args.step in ('components', 'all'):
        compute_coot_components(adapter, results_dir, sigma=args.sigma)


if __name__ == '__main__':
    main()
