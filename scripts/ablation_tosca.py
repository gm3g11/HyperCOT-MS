#!/usr/bin/env python3
"""Parameter ablation study on TOSCA dataset.

Sweeps parameters and evaluates classification accuracy.

Usage:
    python scripts/ablation_tosca.py --sweep alpha
    python scripts/ablation_tosca.py --sweep sigma
    python scripts/ablation_tosca.py --sweep spatial_weight
    python scripts/ablation_tosca.py --sweep mu_type
    python scripts/ablation_tosca.py --sweep all
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mscoot.config import get_results_dir

# Import classification function
from classify_tosca import get_ground_truth, knn_loocv, CLASS_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ablation parameter configurations
SWEEPS = {
    'alpha': {
        'values': [0.0, 0.25, 0.5, 0.75, 1.0],
        'requires_recompute': 'distances',  # only distances
        'default': 0.5,
    },
    'spatial_weight': {
        'values': [0.0, 0.5, 1.0, 2.0],
        'requires_recompute': 'distances',
        'default': 1.0,
    },
    'sigma': {
        'values': [0.1, 0.2, 0.3, 0.5, 1.0],
        'requires_recompute': 'components',  # components + distances
        'default': 0.3,
    },
}


def run_command(cmd, desc=""):
    """Run a shell command and print output."""
    print(f"\n{'─' * 40}")
    print(f"Running: {desc or ' '.join(cmd)}")
    print(f"{'─' * 40}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: Command exited with code {result.returncode}")
    return result.returncode == 0


def sweep_alpha(results_dir):
    """Sweep alpha parameter (only recomputes distances)."""
    values = SWEEPS['alpha']['values']
    results = []
    labels = get_ground_truth()

    for alpha in values:
        print(f"\n{'=' * 60}")
        print(f"Alpha = {alpha}")
        print(f"{'=' * 60}")

        # Recompute FGW
        run_command([
            sys.executable, 'scripts/compute_distances.py',
            '--dataset', 'tosca', '--methods', 'fgw',
            '--alpha', str(alpha),
        ], f"FGW with alpha={alpha}")

        # Recompute MS-COOT
        run_command([
            sys.executable, 'scripts/compute_distances.py',
            '--dataset', 'tosca', '--methods', 'mscoot',
            '--alpha', str(alpha),
        ], f"MS-COOT with alpha={alpha}")

        # Evaluate
        for method, fname in [('fgw', 'fgw_matrix.csv'), ('coot', 'coot_matrix.csv')]:
            mat_path = results_dir / "distance_matrices" / fname
            if mat_path.exists():
                mat = pd.read_csv(mat_path, index_col=0).values
                acc, _, per_class = knn_loocv(mat, labels, k=1)
                results.append({
                    'parameter': 'alpha', 'value': alpha,
                    'method': method, 'k': 1, 'accuracy': acc,
                    **{f'acc_{n}': per_class.get(n, np.nan) for n in CLASS_NAMES},
                })
                print(f"  {method} k=1: {acc:.1%}")

    return results


def sweep_spatial_weight(results_dir):
    """Sweep spatial_weight parameter (only recomputes MS-COOT distances)."""
    values = SWEEPS['spatial_weight']['values']
    results = []
    labels = get_ground_truth()

    for sw in values:
        print(f"\n{'=' * 60}")
        print(f"spatial_weight = {sw}")
        print(f"{'=' * 60}")

        run_command([
            sys.executable, 'scripts/compute_distances.py',
            '--dataset', 'tosca', '--methods', 'mscoot',
            '--spatial-weight', str(sw),
        ], f"MS-COOT with spatial_weight={sw}")

        mat_path = results_dir / "distance_matrices" / "coot_matrix.csv"
        if mat_path.exists():
            mat = pd.read_csv(mat_path, index_col=0).values
            acc, _, per_class = knn_loocv(mat, labels, k=1)
            results.append({
                'parameter': 'spatial_weight', 'value': sw,
                'method': 'coot', 'k': 1, 'accuracy': acc,
                **{f'acc_{n}': per_class.get(n, np.nan) for n in CLASS_NAMES},
            })
            print(f"  MS-COOT k=1: {acc:.1%}")

    return results


def sweep_sigma(results_dir):
    """Sweep sigma parameter (requires full recompute)."""
    values = SWEEPS['sigma']['values']
    results = []
    labels = get_ground_truth()

    for sigma in values:
        print(f"\n{'=' * 60}")
        print(f"sigma = {sigma}")
        print(f"{'=' * 60}")

        # Recompute components
        run_command([
            sys.executable, 'scripts/compute_components.py',
            '--dataset', 'tosca', '--sigma', str(sigma),
        ], f"Components with sigma={sigma}")

        # Recompute all distances
        run_command([
            sys.executable, 'scripts/compute_distances.py',
            '--dataset', 'tosca', '--methods', 'wd,gwd,fgw',
        ], f"WD/GWD/FGW with sigma={sigma}")

        run_command([
            sys.executable, 'scripts/compute_distances.py',
            '--dataset', 'tosca', '--methods', 'mscoot',
        ], f"MS-COOT with sigma={sigma}")

        # Evaluate all methods
        for method, fname in [('wd', 'wd_matrix.csv'), ('gwd', 'gwd_matrix.csv'),
                               ('fgw', 'fgw_matrix.csv'), ('coot', 'coot_matrix.csv')]:
            mat_path = results_dir / "distance_matrices" / fname
            if mat_path.exists():
                mat = pd.read_csv(mat_path, index_col=0).values
                acc, _, per_class = knn_loocv(mat, labels, k=1)
                results.append({
                    'parameter': 'sigma', 'value': sigma,
                    'method': method, 'k': 1, 'accuracy': acc,
                    **{f'acc_{n}': per_class.get(n, np.nan) for n in CLASS_NAMES},
                })
                print(f"  {method} k=1: {acc:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description='TOSCA ablation study')
    parser.add_argument('--sweep', required=True,
                        choices=['alpha', 'sigma', 'spatial_weight', 'all'],
                        help='Parameter to sweep')
    args = parser.parse_args()

    results_dir = get_results_dir('tosca')
    ablation_dir = results_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    if args.sweep in ('alpha', 'all'):
        all_results.extend(sweep_alpha(results_dir))

    if args.sweep in ('spatial_weight', 'all'):
        all_results.extend(sweep_spatial_weight(results_dir))

    if args.sweep in ('sigma', 'all'):
        all_results.extend(sweep_sigma(results_dir))

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        out_path = ablation_dir / f"ablation_{args.sweep}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nAblation results saved to {out_path}")

        # Print summary table
        print(f"\n{'=' * 60}")
        print("Ablation Summary")
        print(f"{'=' * 60}")
        for param in df['parameter'].unique():
            sub = df[df['parameter'] == param]
            print(f"\n{param}:")
            for _, row in sub.iterrows():
                print(f"  {row['value']:>6} | {row['method']:>6} | acc={row['accuracy']:.1%}")


if __name__ == '__main__':
    main()
