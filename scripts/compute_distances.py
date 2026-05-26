#!/usr/bin/env python3
"""Compute distance matrices (WD, GWD, FGW, MS-COOT) for a dataset."""

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mscoot.config import load_dataset_config, get_dataset_dir, get_results_dir
from mscoot.data_adapter import DatasetAdapter
from mscoot.graph.adjacency import build_adjacency_from_edges, compute_graph_distances
from mscoot.solvers.wasserstein import compute_wd
from mscoot.solvers.gromov import compute_gwd
from mscoot.solvers.fused_gw import compute_fgw, build_feature_cost
from mscoot.solvers.mscoot_solver import compute_mscoot

# Shared data for multiprocessing workers (populated before Pool creation, inherited via fork)
_shared = {}


def _limit_blas_threads():
    """Limit BLAS to 1 thread — better for many small parallel GW/FGW calls."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'


def _gwd_pair(ij):
    i, j = ij
    _, dist = compute_gwd(_shared['D'][i], _shared['D'][j])
    return (i, j, dist)


def _fgw_pair(ij):
    i, j = ij
    coords1, C1 = _shared['data'][i]
    coords2, C2 = _shared['data'][j]
    M = build_feature_cost(coords1, coords2, _shared['diagonal'])
    _, dist = compute_fgw(M, C1, C2, alpha=_shared['alpha'])
    return (i, j, dist)


def _get_coord_cols(adapter):
    """Get coordinate column names based on dataset dimensionality."""
    if adapter.config.n_dimensions >= 3:
        return ['x', 'y', 'z']
    return ['x', 'y']


def compute_wd_matrix(adapter, results_dir):
    """Compute all-pairs WD distance matrix."""
    timesteps = adapter.get_timesteps()
    n = len(timesteps)
    diagonal = adapter.get_domain_diagonal()
    coord_cols = _get_coord_cols(adapter)

    # Load all coordinates
    coords_by_ts = {}
    for ts in timesteps:
        cp = adapter.load_critical_points(ts)
        coords_by_ts[ts] = cp[coord_cols].values

    print(f"Computing {n}x{n} WD matrix (diagonal={diagonal:.1f})...")
    matrix = np.zeros((n, n))
    total = n * (n - 1) // 2
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            _, dist = compute_wd(coords_by_ts[timesteps[i]],
                                 coords_by_ts[timesteps[j]], diagonal)
            matrix[i, j] = dist
            matrix[j, i] = dist
            count += 1
            if count % 500 == 0:
                print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")

    out_path = results_dir / "distance_matrices" / "wd_matrix.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=timesteps, columns=timesteps).to_csv(out_path)
    print(f"Saved WD matrix to {out_path}")
    return matrix


def compute_gwd_matrix(adapter, results_dir, workers=1):
    """Compute all-pairs GWD distance matrix."""
    timesteps = adapter.get_timesteps()
    n = len(timesteps)

    coord_cols = _get_coord_cols(adapter)

    # Precompute graph distances for all timesteps
    print(f"Precomputing graph distances for {n} timesteps...")
    D_by_ts = {}
    for ts in tqdm(timesteps, desc="Graph distances"):
        cp = adapter.load_critical_points(ts)
        edges = adapter.load_edges(ts)
        coords = cp[coord_cols].values
        adj = build_adjacency_from_edges(len(cp), edges)
        D_by_ts[ts] = compute_graph_distances(adj, coords)

    print(f"Computing {n}x{n} GWD matrix ({workers} workers)...")
    matrix = np.zeros((n, n))
    total = n * (n - 1) // 2
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    if workers > 1:
        global _shared
        _shared['D'] = [D_by_ts[ts] for ts in timesteps]
        _limit_blas_threads()
        ctx = mp.get_context('fork')
        with ctx.Pool(workers, initializer=_limit_blas_threads) as pool:
            for i, j, dist in tqdm(pool.imap_unordered(_gwd_pair, pairs, chunksize=20),
                                   total=total, desc="Computing GWD"):
                matrix[i, j] = dist
                matrix[j, i] = dist
        _shared.clear()
    else:
        with tqdm(total=total, desc="Computing GWD") as pbar:
            for i, j in pairs:
                _, dist = compute_gwd(D_by_ts[timesteps[i]], D_by_ts[timesteps[j]])
                matrix[i, j] = dist
                matrix[j, i] = dist
                pbar.update(1)

    out_path = results_dir / "distance_matrices" / "gwd_matrix.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=timesteps, columns=timesteps).to_csv(out_path)
    print(f"Saved GWD matrix to {out_path}")
    return matrix


def compute_fgw_matrix(adapter, results_dir, alpha=0.5, workers=1):
    """Compute all-pairs FGW distance matrix."""
    timesteps = adapter.get_timesteps()
    n = len(timesteps)
    diagonal = adapter.get_domain_diagonal()

    coord_cols = _get_coord_cols(adapter)

    # Precompute structure costs
    print(f"Precomputing structure costs for {n} timesteps...")
    data_by_ts = {}
    for ts in tqdm(timesteps, desc="Structure costs"):
        cp = adapter.load_critical_points(ts)
        edges = adapter.load_edges(ts)
        coords = cp[coord_cols].values
        adj = build_adjacency_from_edges(len(cp), edges)
        D = compute_graph_distances(adj, coords)
        data_by_ts[ts] = (coords, D)

    print(f"Computing {n}x{n} FGW matrix (alpha={alpha}, {workers} workers)...")
    matrix = np.zeros((n, n))
    total = n * (n - 1) // 2
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    if workers > 1:
        global _shared
        _shared['data'] = [data_by_ts[ts] for ts in timesteps]
        _shared['diagonal'] = diagonal
        _shared['alpha'] = alpha
        _limit_blas_threads()
        ctx = mp.get_context('fork')
        with ctx.Pool(workers, initializer=_limit_blas_threads) as pool:
            for i, j, dist in tqdm(pool.imap_unordered(_fgw_pair, pairs, chunksize=20),
                                   total=total, desc="Computing FGW"):
                matrix[i, j] = dist
                matrix[j, i] = dist
        _shared.clear()
    else:
        with tqdm(total=total, desc="Computing FGW") as pbar:
            for i, j in pairs:
                coords1, C1 = data_by_ts[timesteps[i]]
                coords2, C2 = data_by_ts[timesteps[j]]
                M = build_feature_cost(coords1, coords2, diagonal)
                _, dist = compute_fgw(M, C1, C2, alpha=alpha)
                matrix[i, j] = dist
                matrix[j, i] = dist
                pbar.update(1)

    out_path = results_dir / "distance_matrices" / "fgw_matrix.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=timesteps, columns=timesteps).to_csv(out_path)
    print(f"Saved FGW matrix to {out_path}")
    return matrix


def load_coot_data(coot_dir, ts, adapter=None):
    """Load precomputed mu, nu, omega for a timestep."""
    mu_df = pd.read_csv(coot_dir / f"mu_{ts:03d}.csv")
    nu_df = pd.read_csv(coot_dir / f"nu_{ts:03d}.csv")
    omega_df = pd.read_csv(coot_dir / f"omega_{ts:03d}.csv", index_col=0)
    result = {
        'mu': mu_df['mu'].values,
        'nu': nu_df['nu'].values,
        'omega': omega_df.values,
        'cp_types': mu_df['cp_type'].values,
    }
    if adapter is not None:
        coord_cols = _get_coord_cols(adapter)
        cp = adapter.load_critical_points(ts)
        result['coords'] = cp[coord_cols].values
    return result


def compute_coot_matrix(adapter, results_dir, alpha=0.5, epsilon=0.001,
                        spatial_weight=1.0):
    """Compute all-pairs MS-COOT distance matrix."""
    coot_dir = results_dir / "coot_data"
    stats = pd.read_csv(coot_dir / "component_stats.csv")
    valid_ts = sorted(stats[stats['status'] == 'ok']['timestep'].values)
    n = len(valid_ts)

    # Preload all data (with coordinates for spatial penalty)
    print(f"Loading COOT components for {n} timesteps...")
    data_by_ts = {}
    for ts in tqdm(valid_ts, desc="Loading"):
        data_by_ts[ts] = load_coot_data(coot_dir, ts, adapter=adapter)

    print(f"Computing {n}x{n} MS-COOT matrix ({n*(n-1)//2} pairs)...")
    print(f"  alpha={alpha}, epsilon={epsilon}, spatial_weight={spatial_weight}")
    matrix = np.zeros((n, n))
    total = n * (n - 1) // 2

    with tqdm(total=total, desc="Computing MS-COOT") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                d1 = data_by_ts[valid_ts[i]]
                d2 = data_by_ts[valid_ts[j]]
                _, _, cost = compute_mscoot(
                    d1['omega'], d2['omega'],
                    d1['mu'], d2['mu'],
                    d1['nu'], d2['nu'],
                    types1=d1['cp_types'], types2=d2['cp_types'],
                    alpha=alpha, epsilon=epsilon,
                    coords1=d1.get('coords'), coords2=d2.get('coords'),
                    spatial_weight=spatial_weight,
                )
                matrix[i, j] = cost
                matrix[j, i] = cost
                pbar.update(1)

    out_path = results_dir / "distance_matrices" / "coot_matrix.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=valid_ts, columns=valid_ts).to_csv(out_path)
    print(f"Saved MS-COOT matrix to {out_path}")
    return matrix


def main():
    parser = argparse.ArgumentParser(description='Compute distance matrices')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--methods', type=str, default='wd,gwd,fgw',
                        help='Comma-separated methods to compute')
    parser.add_argument('--alpha', type=float, default=0.5, help='FGW/COOT alpha')
    parser.add_argument('--epsilon', type=float, default=0.001, help='COOT epsilon')
    parser.add_argument('--spatial-weight', type=float, default=1.0,
                        help='COOT spatial penalty weight')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers for GWD/FGW (default: 1)')
    args = parser.parse_args()

    dataset_dir = get_dataset_dir(args.dataset)
    results_dir = get_results_dir(args.dataset)
    results_dir.mkdir(parents=True, exist_ok=True)

    adapter = DatasetAdapter(dataset_dir)
    methods = [m.strip().lower() for m in args.methods.split(',')]

    if 'wd' in methods:
        compute_wd_matrix(adapter, results_dir)
    if 'gwd' in methods:
        compute_gwd_matrix(adapter, results_dir, workers=args.workers)
    if 'fgw' in methods:
        compute_fgw_matrix(adapter, results_dir, alpha=args.alpha,
                           workers=args.workers)
    if 'mscoot' in methods:
        compute_coot_matrix(adapter, results_dir, alpha=args.alpha,
                            epsilon=args.epsilon,
                            spatial_weight=args.spatial_weight)


if __name__ == '__main__':
    main()
