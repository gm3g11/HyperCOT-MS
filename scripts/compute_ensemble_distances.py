#!/usr/bin/env python3
"""Compute inter-run distance matrices for the viscous finger ensemble.

For each method (WD, GWD, FGW, MS-COOT), computes a 30×30 distance matrix
where entry (i, j) = mean distance over K representative timesteps.

Usage:
    python scripts/compute_ensemble_distances.py --methods wd,gwd,fgw
    python scripts/compute_ensemble_distances.py --methods mscoot --workers 4
    python scripts/compute_ensemble_distances.py --methods wd,gwd,fgw,mscoot --workers 8
"""

import argparse
import multiprocessing as mp
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mscoot.config import load_dataset_config, get_dataset_dir, get_results_dir
from mscoot.data_adapter import DatasetAdapter
from mscoot.graph.adjacency import build_adjacency_from_edges, compute_graph_distances
from mscoot.solvers.wasserstein import compute_wd
from mscoot.solvers.gromov import compute_gwd
from mscoot.solvers.fused_gw import compute_fgw, build_feature_cost
from mscoot.solvers.mscoot_solver import compute_mscoot

# Shared data for multiprocessing workers
_shared = {}


def _limit_blas_threads():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'


def get_project_root():
    return Path(__file__).resolve().parent.parent


def load_ensemble_config(ensemble_path):
    with open(ensemble_path) as f:
        return yaml.safe_load(f)


def get_dataset_list(config, runs=None):
    """Get ordered list of (dataset_name, class_name, run_id)."""
    selected_runs = runs or config['selected_runs']
    datasets = []
    for class_name in ['coarse', 'medium', 'fine']:
        class_info = config['classes'][class_name]
        prefix = class_info['prefix']
        for run_id in selected_runs:
            datasets.append((f"{prefix}_{run_id}", class_name, run_id))
    return datasets


def _get_coord_cols(adapter):
    if adapter.config.n_dimensions >= 3:
        return ['x', 'y', 'z']
    return ['x', 'y']


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


def preload_all_data(datasets, rep_timesteps, methods):
    """Preload all data needed for distance computation.

    Returns:
        data: dict mapping (dataset_name, ts) -> dict with keys depending on methods
    """
    data = {}
    need_coords = any(m in methods for m in ['wd', 'fgw', 'mscoot'])
    need_graph = any(m in methods for m in ['gwd', 'fgw'])
    need_coot = 'mscoot' in methods

    for dataset_name, class_name, run_id in tqdm(datasets, desc="Loading data"):
        dataset_dir = get_dataset_dir(dataset_name)
        results_dir = get_results_dir(dataset_name)
        coot_dir = results_dir / "coot_data"

        ttk_dir = dataset_dir / "ttk_output"
        if not dataset_dir.exists() or not ttk_dir.exists():
            print(f"  WARNING: {dataset_name} not found or no TTK output, skipping")
            continue

        adapter = DatasetAdapter(dataset_dir)
        coord_cols = _get_coord_cols(adapter)
        diagonal = adapter.get_domain_diagonal()

        # Check which timesteps have valid COOT data
        stats_path = coot_dir / "component_stats.csv"
        valid_ts = set()
        if stats_path.exists():
            stats = pd.read_csv(stats_path)
            valid_ts = set(stats[stats['status'] == 'ok']['timestep'].values)

        for ts in rep_timesteps:
            entry = {'diagonal': diagonal}

            # Load CPs and coordinates
            try:
                cp = adapter.load_critical_points(ts)
            except Exception as e:
                print(f"  WARNING: {dataset_name} ts={ts}: {e}")
                continue

            if need_coords:
                entry['coords'] = cp[coord_cols].values

            if need_graph:
                edges = adapter.load_edges(ts)
                adj = build_adjacency_from_edges(len(cp), edges)
                entry['D'] = compute_graph_distances(adj, cp[coord_cols].values)

            if need_coot and ts in valid_ts:
                try:
                    coot_data = load_coot_data(coot_dir, ts, adapter=adapter)
                    entry.update(coot_data)
                    entry['has_coot'] = True
                except Exception as e:
                    print(f"  WARNING: {dataset_name} ts={ts} COOT: {e}")
                    entry['has_coot'] = False
            else:
                entry['has_coot'] = False

            data[(dataset_name, ts)] = entry

    return data


def _gwd_worker(args):
    """Worker for parallel GWD computation."""
    i, j, ts = args
    key_i = _shared['keys'][i]
    key_j = _shared['keys'][j]
    d_i = _shared['data'].get((key_i, ts))
    d_j = _shared['data'].get((key_j, ts))
    if d_i is None or d_j is None or 'D' not in d_i or 'D' not in d_j:
        return (i, j, ts, np.nan)
    _, dist = compute_gwd(d_i['D'], d_j['D'])
    return (i, j, ts, dist)


def _fgw_worker(args):
    """Worker for parallel FGW computation."""
    i, j, ts = args
    key_i = _shared['keys'][i]
    key_j = _shared['keys'][j]
    d_i = _shared['data'].get((key_i, ts))
    d_j = _shared['data'].get((key_j, ts))
    if (d_i is None or d_j is None
            or 'D' not in d_i or 'D' not in d_j
            or 'coords' not in d_i or 'coords' not in d_j):
        return (i, j, ts, np.nan)
    diag = max(d_i['diagonal'], d_j['diagonal'])
    M = build_feature_cost(d_i['coords'], d_j['coords'], diag)
    _, dist = compute_fgw(M, d_i['D'], d_j['D'], alpha=_shared['alpha'])
    return (i, j, ts, dist)


def _coot_worker(args):
    """Worker for parallel MS-COOT computation."""
    i, j, ts = args
    key_i = _shared['keys'][i]
    key_j = _shared['keys'][j]
    d_i = _shared['data'].get((key_i, ts))
    d_j = _shared['data'].get((key_j, ts))
    if d_i is None or d_j is None:
        return (i, j, ts, np.nan)
    if not d_i.get('has_coot') or not d_j.get('has_coot'):
        return (i, j, ts, np.nan)
    try:
        _, _, cost = compute_mscoot(
            d_i['omega'], d_j['omega'],
            d_i['mu'], d_j['mu'],
            d_i['nu'], d_j['nu'],
            types1=d_i['cp_types'], types2=d_j['cp_types'],
            alpha=_shared['alpha'], epsilon=_shared['epsilon'],
            coords1=d_i.get('coords'), coords2=d_j.get('coords'),
            spatial_weight=_shared['spatial_weight'],
        )
        return (i, j, ts, cost)
    except Exception as e:
        return (i, j, ts, np.nan)


def compute_inter_run_matrix(method, data, dataset_names, rep_timesteps,
                             workers=1, alpha=0.5, epsilon=0.001,
                             spatial_weight=1.0):
    """Compute inter-run distance matrix for a single method.

    Args:
        method: 'wd', 'gwd', 'fgw', or 'mscoot'
        data: preloaded data dict from preload_all_data
        dataset_names: list of dataset names (length N)
        rep_timesteps: list of representative timesteps
        workers: number of parallel workers

    Returns:
        N×N numpy distance matrix
    """
    N = len(dataset_names)
    K = len(rep_timesteps)

    # Accumulate per-timestep distances
    dist_sum = np.zeros((N, N))
    dist_count = np.zeros((N, N))

    pairs_ts = [(i, j, ts)
                for i in range(N) for j in range(i + 1, N)
                for ts in rep_timesteps]
    total = len(pairs_ts)

    print(f"\nComputing {method.upper()} inter-run distances "
          f"({N}×{N}, K={K}, {total} pairs, {workers} workers)...")

    if method == 'wd':
        # WD is fast, run serial
        for i, j, ts in tqdm(pairs_ts, desc="WD"):
            d_i = data.get((dataset_names[i], ts))
            d_j = data.get((dataset_names[j], ts))
            if d_i is None or d_j is None:
                continue
            if 'coords' not in d_i or 'coords' not in d_j:
                continue
            diag = max(d_i['diagonal'], d_j['diagonal'])
            _, dist = compute_wd(d_i['coords'], d_j['coords'], diag)
            dist_sum[i, j] += dist
            dist_sum[j, i] += dist
            dist_count[i, j] += 1
            dist_count[j, i] += 1

    elif method in ('gwd', 'fgw', 'mscoot'):
        worker_fn = {'gwd': _gwd_worker, 'fgw': _fgw_worker,
                     'mscoot': _coot_worker}[method]

        global _shared
        _shared['data'] = data
        _shared['keys'] = dataset_names
        _shared['alpha'] = alpha
        _shared['epsilon'] = epsilon
        _shared['spatial_weight'] = spatial_weight

        if workers > 1:
            _limit_blas_threads()
            ctx = mp.get_context('fork')
            with ctx.Pool(workers, initializer=_limit_blas_threads) as pool:
                for i, j, ts, dist in tqdm(
                        pool.imap_unordered(worker_fn, pairs_ts, chunksize=10),
                        total=total, desc=method.upper()):
                    if not np.isnan(dist):
                        dist_sum[i, j] += dist
                        dist_sum[j, i] += dist
                        dist_count[i, j] += 1
                        dist_count[j, i] += 1
        else:
            for args in tqdm(pairs_ts, desc=method.upper()):
                i, j, ts, dist = worker_fn(args)
                if not np.isnan(dist):
                    dist_sum[i, j] += dist
                    dist_sum[j, i] += dist
                    dist_count[i, j] += 1
                    dist_count[j, i] += 1

        _shared.clear()

    # Average over timesteps
    matrix = np.zeros((N, N))
    valid = dist_count > 0
    matrix[valid] = dist_sum[valid] / dist_count[valid]

    # Report coverage
    expected = N * (N - 1) // 2
    actual = np.sum(dist_count[np.triu_indices(N, k=1)] > 0)
    print(f"  Coverage: {int(actual)}/{expected} pairs "
          f"({100*actual/expected:.0f}%)")

    return matrix


def main():
    parser = argparse.ArgumentParser(
        description='Compute inter-run distance matrices for VF ensemble')
    parser.add_argument('--ensemble', type=str, default=None,
                        help='Path to ensemble.yaml')
    parser.add_argument('--methods', type=str, default='wd,gwd,fgw,mscoot',
                        help='Comma-separated methods (default: wd,gwd,fgw,mscoot)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers for GWD/FGW/MS-COOT')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='FGW/COOT alpha (default: 0.5)')
    parser.add_argument('--epsilon', type=float, default=0.001,
                        help='COOT epsilon (default: 0.001)')
    parser.add_argument('--spatial-weight', type=float, default=1.0,
                        help='COOT spatial penalty weight (default: 1.0)')
    parser.add_argument('--runs', type=str, default=None,
                        help='Comma-separated run IDs (default: all selected)')
    args = parser.parse_args()

    root = get_project_root()
    ensemble_path = Path(args.ensemble) if args.ensemble else \
        root / "datasets" / "vf_ensemble" / "ensemble.yaml"
    config = load_ensemble_config(ensemble_path)

    runs = args.runs.split(',') if args.runs else None
    methods = [m.strip().lower() for m in args.methods.split(',')]
    rep_timesteps = config['representative_timesteps']

    datasets = get_dataset_list(config, runs=runs)
    dataset_names = [d[0] for d in datasets]
    class_labels = [d[1] for d in datasets]
    N = len(datasets)

    print(f"Ensemble: {N} datasets, K={len(rep_timesteps)} timesteps")
    print(f"Methods:  {', '.join(methods)}")
    print(f"Workers:  {args.workers}")
    print()

    # Preload all data
    data = preload_all_data(datasets, rep_timesteps, methods)
    print(f"Loaded data for {len(data)} (dataset, timestep) pairs")

    # Output directory
    out_dir = root / "results" / "vf_ensemble" / "inter_run_distances"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute distance matrices
    for method in methods:
        matrix = compute_inter_run_matrix(
            method, data, dataset_names, rep_timesteps,
            workers=args.workers, alpha=args.alpha, epsilon=args.epsilon,
            spatial_weight=args.spatial_weight,
        )

        # Save matrix
        method_label = 'coot' if method == 'mscoot' else method
        out_path = out_dir / f"{method_label}_matrix.csv"
        df = pd.DataFrame(matrix, index=dataset_names, columns=dataset_names)
        df.to_csv(out_path)
        print(f"  Saved {out_path}")

    # Save metadata
    meta = {
        'dataset_names': dataset_names,
        'class_labels': class_labels,
        'representative_timesteps': rep_timesteps,
    }
    meta_path = out_dir / "metadata.csv"
    pd.DataFrame({
        'dataset': dataset_names,
        'class': class_labels,
        'run_id': [d[2] for d in datasets],
    }).to_csv(meta_path, index=False)
    print(f"  Saved {meta_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
