#!/usr/bin/env python3
"""Classify viscous finger ensemble runs by resolution using LOO 1-NN.

Reads 30×30 inter-run distance matrices and evaluates classification
accuracy with permutation test for statistical significance.

Usage:
    python scripts/classify_ensemble.py
    python scripts/classify_ensemble.py --n-permutations 5000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def get_project_root():
    return Path(__file__).resolve().parent.parent


def loo_1nn(distance_matrix, labels):
    """Leave-one-out 1-nearest-neighbor classification.

    Args:
        distance_matrix: N×N symmetric distance matrix (zero diagonal)
        labels: length-N array of class labels

    Returns:
        accuracy: fraction correctly classified
        predictions: length-N array of predicted labels
        correct: boolean array of correct predictions
    """
    N = len(labels)
    predictions = []

    for i in range(N):
        # Find nearest neighbor (excluding self)
        dists = distance_matrix[i].copy()
        dists[i] = np.inf  # Exclude self
        nn_idx = np.argmin(dists)
        predictions.append(labels[nn_idx])

    predictions = np.array(predictions)
    correct = predictions == labels
    accuracy = np.mean(correct)

    return accuracy, predictions, correct


def permutation_test(distance_matrix, labels, n_permutations=1000,
                     rng=None):
    """Permutation test for classification significance.

    Shuffles class labels and recomputes LOO 1-NN accuracy to build
    null distribution. p-value = fraction of permuted accuracies >= observed.

    Args:
        distance_matrix: N×N distance matrix
        labels: true class labels
        n_permutations: number of permutations
        rng: numpy random generator

    Returns:
        p_value: significance level
        null_accuracies: array of permuted accuracies
    """
    if rng is None:
        rng = np.random.default_rng(42)

    observed_acc, _, _ = loo_1nn(distance_matrix, labels)
    null_accuracies = np.zeros(n_permutations)

    for p in range(n_permutations):
        perm_labels = rng.permutation(labels)
        null_accuracies[p], _, _ = loo_1nn(distance_matrix, perm_labels)

    p_value = np.mean(null_accuracies >= observed_acc)
    return p_value, null_accuracies


def confusion_matrix(labels, predictions, class_names=None):
    """Compute confusion matrix.

    Returns:
        cm: K×K matrix where cm[i,j] = count of true class i predicted as j
        class_names: ordered class names
    """
    if class_names is None:
        class_names = sorted(set(labels))

    K = len(class_names)
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    cm = np.zeros((K, K), dtype=int)

    for true, pred in zip(labels, predictions):
        cm[name_to_idx[true], name_to_idx[pred]] += 1

    return cm, class_names


def main():
    parser = argparse.ArgumentParser(
        description='Classify VF ensemble runs by resolution (LOO 1-NN)')
    parser.add_argument('--ensemble', type=str, default=None,
                        help='Path to ensemble.yaml')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Path to inter_run_distances/ directory')
    parser.add_argument('--methods', type=str, default='wd,gwd,fgw,coot',
                        help='Comma-separated method names (matching matrix filenames)')
    parser.add_argument('--n-permutations', type=int, default=1000,
                        help='Number of permutations for significance test')
    args = parser.parse_args()

    root = get_project_root()
    results_dir = Path(args.results_dir) if args.results_dir else \
        root / "results" / "vf_ensemble" / "inter_run_distances"

    # Load metadata
    meta_path = results_dir / "metadata.csv"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run compute_ensemble_distances.py first.")
        sys.exit(1)

    meta = pd.read_csv(meta_path)
    labels = meta['class'].values
    dataset_names = meta['dataset'].values
    N = len(labels)

    class_names = ['coarse', 'medium', 'fine']
    class_counts = {c: np.sum(labels == c) for c in class_names}
    random_baseline = 1.0 / len(class_names)

    print(f"Classification: LOO 1-NN on {N} samples")
    print(f"Classes: {class_counts}")
    print(f"Random baseline: {random_baseline:.1%}")
    print(f"Permutations: {args.n_permutations}")
    print()

    methods = [m.strip() for m in args.methods.split(',')]
    all_results = []

    print(f"{'Method':<12} {'Accuracy':>10} {'Correct':>10} {'p-value':>10}")
    print("-" * 45)

    for method in methods:
        matrix_path = results_dir / f"{method}_matrix.csv"
        if not matrix_path.exists():
            print(f"{method:<12} {'MISSING':>10}")
            continue

        # Load distance matrix
        df = pd.read_csv(matrix_path, index_col=0)
        matrix = df.values

        # Verify symmetry and zero diagonal
        assert matrix.shape == (N, N), \
            f"Matrix shape {matrix.shape} != ({N}, {N})"
        assert np.allclose(matrix, matrix.T, atol=1e-10), \
            f"{method} matrix not symmetric"

        # LOO 1-NN classification
        accuracy, predictions, correct = loo_1nn(matrix, labels)
        n_correct = int(np.sum(correct))

        # Permutation test
        p_value, null_dist = permutation_test(
            matrix, labels, n_permutations=args.n_permutations)

        # Significance marker
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = ""

        print(f"{method:<12} {accuracy:>8.1%}   {n_correct:>3}/{N}     "
              f"p={p_value:.4f} {sig}")

        # Confusion matrix
        cm, cn = confusion_matrix(labels, predictions, class_names)

        all_results.append({
            'method': method,
            'accuracy': accuracy,
            'n_correct': n_correct,
            'n_total': N,
            'p_value': p_value,
            'confusion_matrix': cm,
        })

    # Print confusion matrices
    print()
    for result in all_results:
        method = result['method']
        cm = result['confusion_matrix']
        print(f"Confusion matrix ({method}, acc={result['accuracy']:.1%}):")
        print(f"  {'':>12} {'coarse':>8} {'medium':>8} {'fine':>8}")
        for i, cn in enumerate(class_names):
            print(f"  {cn:>12} {cm[i,0]:>8} {cm[i,1]:>8} {cm[i,2]:>8}")
        print()

    # Save results
    out_dir = root / "results" / "vf_ensemble" / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for r in all_results:
        summary_rows.append({
            'method': r['method'],
            'accuracy': r['accuracy'],
            'n_correct': r['n_correct'],
            'n_total': r['n_total'],
            'p_value': r['p_value'],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "classification_results.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")

    # Save confusion matrices
    for r in all_results:
        cm_df = pd.DataFrame(r['confusion_matrix'],
                             index=class_names, columns=class_names)
        cm_path = out_dir / f"confusion_{r['method']}.csv"
        cm_df.to_csv(cm_path)

    # Save per-sample predictions
    for r in all_results:
        method = r['method']
        matrix_path = results_dir / f"{method}_matrix.csv"
        df = pd.read_csv(matrix_path, index_col=0)
        _, predictions, correct = loo_1nn(df.values, labels)
        pred_df = pd.DataFrame({
            'dataset': dataset_names,
            'true_class': labels,
            'predicted_class': predictions,
            'correct': correct,
        })
        pred_path = out_dir / f"predictions_{method}.csv"
        pred_df.to_csv(pred_path, index=False)


if __name__ == '__main__':
    main()
