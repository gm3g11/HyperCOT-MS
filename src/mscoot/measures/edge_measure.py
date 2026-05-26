"""Hyperedge measure (nu) computation."""

import numpy as np
import pandas as pd
from typing import List, Tuple


def compute_nu(mu: np.ndarray, hyper_df: pd.DataFrame) -> Tuple[np.ndarray, List[List[int]]]:
    """Compute hyperedge measure nu as sum of boundary CP mu values.

    Args:
        mu: Node measure array (n_cp,).
        hyper_df: Hypergraph DataFrame with 'hyperedge' column (str repr of member list).

    Returns:
        Tuple of (nu array, list of member lists).
        nu is normalized to sum to 1.
    """
    n_hyper = len(hyper_df)
    if n_hyper == 0:
        return np.array([]), []

    nu_unnorm = np.zeros(n_hyper)
    hyperedge_members = []

    for idx, row in hyper_df.iterrows():
        members = eval(row['hyperedge'])
        hyperedge_members.append(members)
        nu_unnorm[idx] = sum(mu[cp] for cp in members if cp < len(mu))

    total = nu_unnorm.sum()
    if total > 1e-9:
        nu = nu_unnorm / total
    else:
        nu = np.ones(n_hyper) / n_hyper

    return nu, hyperedge_members


def recompute_nu(mu: np.ndarray, hyper_df: pd.DataFrame) -> np.ndarray:
    """Recompute nu from updated mu (e.g., after applying floor).

    Args:
        mu: Updated node measure.
        hyper_df: Hypergraph DataFrame.

    Returns:
        Updated nu, normalized to sum to 1.
    """
    nu, _ = compute_nu(mu, hyper_df)
    return nu
