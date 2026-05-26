#!/usr/bin/env python3
"""k-NN LOOCV classification on TOSCA distance matrices.

Evaluates shape classification accuracy using distance matrices from
WD, GWD, FGW, and MS-COOT methods.

Usage:
    python scripts/classify_tosca.py
    python scripts/classify_tosca.py --k 1,3,5 --methods wd,gwd,fgw,coot
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mscoot.config import get_results_dir

# Ground truth: natural sort order → class labels
# 80 meshes total, 9 classes
CLASS_NAMES = ['cat', 'centaur', 'david', 'dog', 'gorilla',
               'horse', 'michael', 'victoria', 'wolf']

CLASS_RANGES = {
    'cat': (0, 11),       # 11 meshes
    'centaur': (11, 17),  # 6
    'david': (17, 24),    # 7
    'dog': (24, 33),      # 9
    'gorilla': (33, 37),  # 4
    'horse': (37, 45),    # 8
    'michael': (45, 65),  # 20
    'victoria': (65, 77), # 12
    'wolf': (77, 80),     # 3
}


def get_ground_truth(n=80):
    """Return ground truth class labels for the 80 TOSCA meshes."""
    labels = np.zeros(n, dtype=int)
    for class_idx, (name, (start, end)) in enumerate(CLASS_RANGES.items()):
        labels[start:end] = class_idx
    return labels


def knn_loocv(dist_matrix, labels, k=1):
    """Leave-one-out cross-validation with k-NN on a distance matrix.

    Args:
        dist_matrix: (n, n) symmetric distance matrix
        labels: (n,) integer class labels
        k: number of neighbors

    Returns:
        accuracy: overall accuracy
        predictions: (n,) predicted labels
        per_class_acc: dict of class_name -> accuracy
    """
    n = len(labels)
    predictions = np.zeros(n, dtype=int)

    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf  # exclude self
        nn_indices = np.argsort(dists)[:k]
        nn_labels = labels[nn_indices]
        # Majority vote (break ties by closest neighbor)
        counts = Counter(nn_labels)
        predictions[i] = counts.most_common(1)[0][0]

    accuracy = (predictions == labels).mean()

    # Per-class accuracy
    per_class_acc = {}
    for class_idx, name in enumerate(CLASS_NAMES):
        mask = labels == class_idx
        if mask.sum() > 0:
            per_class_acc[name] = (predictions[mask] == labels[mask]).mean()

    return accuracy, predictions, per_class_acc


def compute_confusion_matrix(labels, predictions, n_classes=9):
    """Compute confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(labels, predictions):
        cm[true, pred] += 1
    return cm


def main():
    parser = argparse.ArgumentParser(description='k-NN LOOCV classification on TOSCA')
    parser.add_argument('--k', type=str, default='1,3,5',
                        help='Comma-separated k values (default: 1,3,5)')
    parser.add_argument('--methods', type=str, default='wd,gwd,fgw,coot',
                        help='Comma-separated methods (default: wd,gwd,fgw,coot)')
    args = parser.parse_args()

    results_dir = get_results_dir('tosca')
    dist_dir = results_dir / "distance_matrices"
    out_dir = results_dir / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)

    k_values = [int(k) for k in args.k.split(',')]
    methods = [m.strip() for m in args.methods.split(',')]

    labels = get_ground_truth()
    n = len(labels)

    print(f"TOSCA Classification: {n} meshes, {len(CLASS_NAMES)} classes")
    print(f"Class distribution: {dict(zip(CLASS_NAMES, [int((labels == i).sum()) for i in range(len(CLASS_NAMES))]))}")
    print(f"Methods: {methods}")
    print(f"k values: {k_values}")
    print("=" * 70)

    # Map method names to matrix filenames
    matrix_names = {
        'wd': 'wd_matrix.csv',
        'gwd': 'gwd_matrix.csv',
        'fgw': 'fgw_matrix.csv',
        'coot': 'coot_matrix.csv',
        'mscoot': 'coot_matrix.csv',
    }

    results = []
    best_method = None
    best_acc = -1

    for method in methods:
        fname = matrix_names.get(method, f'{method}_matrix.csv')
        mat_path = dist_dir / fname
        if not mat_path.exists():
            print(f"\n{method.upper()}: SKIPPED (no {fname})")
            continue

        mat_df = pd.read_csv(mat_path, index_col=0)
        dist_matrix = mat_df.values

        # Validate matrix
        assert dist_matrix.shape[0] == dist_matrix.shape[1], f"Matrix not square: {dist_matrix.shape}"
        actual_n = dist_matrix.shape[0]
        if actual_n != n:
            print(f"\n{method.upper()}: Matrix is {actual_n}x{actual_n}, expected {n}x{n}")
            # Adjust labels if needed (some timesteps may be missing)
            ts_indices = mat_df.index.astype(int).values
            method_labels = labels[ts_indices] if max(ts_indices) < n else labels[:actual_n]
        else:
            method_labels = labels

        print(f"\n{method.upper()} ({actual_n}x{actual_n}):")
        print(f"  Symmetry check: max|D-D'|={np.max(np.abs(dist_matrix - dist_matrix.T)):.2e}")
        print(f"  Range: [{dist_matrix[dist_matrix > 0].min():.4f}, {dist_matrix.max():.4f}]")

        for k in k_values:
            acc, preds, per_class = knn_loocv(dist_matrix, method_labels, k=k)
            results.append({
                'method': method,
                'k': k,
                'accuracy': acc,
                **{f'acc_{name}': per_class.get(name, np.nan) for name in CLASS_NAMES},
            })

            per_class_str = ", ".join(f"{name}={per_class.get(name, 0):.0%}"
                                       for name in CLASS_NAMES)
            print(f"  k={k}: accuracy={acc:.1%} | {per_class_str}")

            if k == 1 and acc > best_acc:
                best_acc = acc
                best_method = method

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "classification_results.csv", index=False)
    print(f"\nResults saved to {out_dir / 'classification_results.csv'}")

    # Save confusion matrix for best method at k=1
    if best_method:
        fname = matrix_names.get(best_method, f'{best_method}_matrix.csv')
        mat_df = pd.read_csv(dist_dir / fname, index_col=0)
        _, preds, _ = knn_loocv(mat_df.values, labels, k=1)
        cm = compute_confusion_matrix(labels, preds)
        cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
        cm_df.to_csv(out_dir / f"confusion_matrix_{best_method}_k1.csv")
        print(f"Confusion matrix saved for {best_method} (k=1, acc={best_acc:.1%})")

    # Summary table
    print(f"\n{'=' * 70}")
    print("Summary (k=1 accuracy):")
    for r in results:
        if r['k'] == 1:
            print(f"  {r['method']:10s}: {r['accuracy']:.1%}")


if __name__ == '__main__':
    main()
