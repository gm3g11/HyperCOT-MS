#!/usr/bin/env python3
"""Batch prepare viscous finger ensemble datasets.

For each run in the ensemble:
  1. Create dataset.yaml (from template in ensemble.yaml)
  2. Run TTK extraction (pvpython_extract_ms.py)
  3. Run compute_components.py (hypergraphs + mu/nu/omega)

Usage:
    # Create dataset.yaml files only (no TTK/components):
    python scripts/prepare_vf_ensemble.py --step yaml

    # Run TTK extraction (requires pvpython):
    /Applications/ParaView-5.13.3.app/Contents/bin/pvpython \
        scripts/prepare_vf_ensemble.py --step ttk

    # Run components (after TTK):
    python scripts/prepare_vf_ensemble.py --step components

    # All steps:
    python scripts/prepare_vf_ensemble.py --step all

    # Subset of runs:
    python scripts/prepare_vf_ensemble.py --step yaml --runs 01,03,05
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def get_project_root():
    return Path(__file__).resolve().parent.parent


def load_ensemble_config(ensemble_path):
    with open(ensemble_path) as f:
        return yaml.safe_load(f)


def get_all_dataset_names(config, runs=None):
    """Get list of (dataset_name, class_name, run_id) tuples."""
    selected_runs = runs or config['selected_runs']
    datasets = []
    for class_name, class_info in config['classes'].items():
        prefix = class_info['prefix']
        for run_id in selected_runs:
            dataset_name = f"{prefix}_{run_id}"
            datasets.append((dataset_name, class_name, run_id))
    return datasets


def create_dataset_yaml(dataset_name, config, root):
    """Create dataset.yaml for a single run from the ensemble template."""
    dataset_dir = root / "datasets" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    template = config['dataset_template']
    dataset_config = {
        'name': dataset_name,
        **template,
    }

    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    return yaml_path


PVPYTHON_DEFAULT = (
    '/Applications/ParaView-5.13.3.app/Contents/bin/pvpython'
)


def run_ttk_extraction(dataset_name, root, threshold=None, pvpython=None):
    """Run pvpython TTK extraction for a dataset.

    TTK scripts require pvpython (ParaView's Python), not regular Python.
    """
    script = root / "scripts" / "pvpython_extract_ms.py"
    mesh_dir = root / "datasets" / dataset_name / "mesh"

    if not mesh_dir.exists() or not list(mesh_dir.glob("*.vti")):
        print(f"  SKIP {dataset_name}: no VTI files in {mesh_dir}")
        return False

    # Use pvpython, not sys.executable
    pv = pvpython or PVPYTHON_DEFAULT
    if not os.path.exists(pv):
        print(f"  ERROR: pvpython not found at {pv}")
        print(f"         Set --pvpython or install ParaView")
        return False

    cmd = [pv, str(script), '--dataset', dataset_name]
    if threshold is not None:
        cmd.extend(['--threshold', str(threshold)])

    print(f"  Running TTK on {dataset_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-500:]}")
        return False

    # Print summary line from output
    for line in result.stdout.split('\n'):
        if 'CPs/timestep' in line or 'Threshold' in line:
            print(f"    {line.strip()}")

    return True


def run_compute_components(dataset_name, root, sigma=0.3):
    """Run compute_components.py for a dataset."""
    script = root / "scripts" / "compute_components.py"
    ttk_dir = root / "datasets" / dataset_name / "ttk_output"

    if not ttk_dir.exists():
        print(f"  SKIP {dataset_name}: no ttk_output/")
        return False

    cmd = [
        sys.executable, str(script),
        '--dataset', dataset_name,
        '--sigma', str(sigma),
    ]

    print(f"  Computing components for {dataset_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-500:]}")
        return False

    # Print summary
    for line in result.stdout.split('\n'):
        if 'Components saved' in line:
            print(f"    {line.strip()}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Batch prepare viscous finger ensemble datasets')
    parser.add_argument('--ensemble', type=str, default=None,
                        help='Path to ensemble.yaml')
    parser.add_argument('--step', choices=['yaml', 'ttk', 'components', 'all'],
                        default='all', help='Which step to run')
    parser.add_argument('--runs', type=str, default=None,
                        help='Comma-separated run IDs (default: all selected)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Fixed persistence threshold for TTK (default: from template)')
    parser.add_argument('--sigma', type=float, default=0.3,
                        help='Persistence image kernel width (default: 0.3)')
    parser.add_argument('--pvpython', type=str, default=None,
                        help=f'Path to pvpython binary (default: {PVPYTHON_DEFAULT})')
    args = parser.parse_args()

    root = get_project_root()
    ensemble_path = Path(args.ensemble) if args.ensemble else \
        root / "datasets" / "vf_ensemble" / "ensemble.yaml"
    config = load_ensemble_config(ensemble_path)

    runs = args.runs.split(',') if args.runs else None
    datasets = get_all_dataset_names(config, runs=runs)

    # Get threshold from template if not specified
    threshold = args.threshold
    if threshold is None:
        threshold = config['dataset_template'].get('parameters', {}).get(
            'persistence_threshold')

    print(f"Ensemble: {len(datasets)} datasets "
          f"({len(config['classes'])} classes × "
          f"{len(runs or config['selected_runs'])} runs)")
    print(f"Step: {args.step}")
    if threshold is not None:
        print(f"Threshold: {threshold}")
    print()

    # Step 1: Create dataset.yaml files
    if args.step in ('yaml', 'all'):
        print("=" * 60)
        print("Creating dataset.yaml files")
        print("=" * 60)
        for dataset_name, class_name, run_id in datasets:
            yaml_path = create_dataset_yaml(dataset_name, config, root)
            print(f"  {dataset_name}: {yaml_path}")
        print()

    # Step 2: TTK extraction
    if args.step in ('ttk', 'all'):
        print("=" * 60)
        print("Running TTK extraction")
        print("=" * 60)
        ok, fail = 0, 0
        for dataset_name, class_name, run_id in datasets:
            if run_ttk_extraction(dataset_name, root, threshold=threshold,
                                  pvpython=args.pvpython):
                ok += 1
            else:
                fail += 1
        print(f"\nTTK: {ok} succeeded, {fail} failed/skipped")
        print()

    # Step 3: Compute components
    if args.step in ('components', 'all'):
        print("=" * 60)
        print("Computing MS-COOT components")
        print("=" * 60)
        ok, fail = 0, 0
        for dataset_name, class_name, run_id in datasets:
            if run_compute_components(dataset_name, root, sigma=args.sigma):
                ok += 1
            else:
                fail += 1
        print(f"\nComponents: {ok} succeeded, {fail} failed/skipped")

    print("\nDone.")


if __name__ == '__main__':
    main()
