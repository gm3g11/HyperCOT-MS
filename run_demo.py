#!/usr/bin/env python3
"""
MS-COOT Demo: Sinusoidal Dataset (Clean vs. Noisy)
===================================================

Demonstrates the full MS-COOT pipeline on the included sinusoidal
example. Compares clean and noisy Morse-Smale complexes using four
methods: WD, GWD, FGW, and MS-COOT.

Usage:
    pip install -e .
    python run_demo.py

No external data download needed — sinusoidal example is included.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure src/ is on the path (works without pip install too)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mscoot.graph.hypergraph import build_hypergraph
from mscoot.graph.adjacency import build_adjacency_from_edges, compute_graph_distances
from mscoot.measures.node_measure import compute_mu
from mscoot.measures.omega import compute_omega
from mscoot.solvers.wasserstein import compute_wd
from mscoot.solvers.gromov import compute_gwd
from mscoot.solvers.fused_gw import compute_fgw, build_feature_cost
from mscoot.solvers.mscoot_solver import compute_mscoot
from mscoot.utils import compute_type_preservation

DATA_DIR = Path(__file__).resolve().parent / "datasets" / "sinusoidal" / "ttk_output"


def load_variant(prefix):
    """Load one variant (clean or noisy) and build all MS-COOT components."""

    # --- Load raw TTK output ---
    cp_raw = pd.read_csv(DATA_DIR / f"{prefix}_critical_points.csv")
    sep_raw = pd.read_csv(DATA_DIR / f"{prefix}_separatrices_cells.csv")
    seg_raw = pd.read_csv(DATA_DIR / f"{prefix}_segmentation.csv")

    # Standardize columns
    cp = pd.DataFrame({
        'x': cp_raw['Points_0'].values,
        'y': cp_raw['Points_1'].values,
        'z': cp_raw['Points_2'].values,
        'cp_type': cp_raw['CellDimension'].values.astype(int),
        'cell_id': cp_raw['CellId'].values.astype(int),
        'scalar': cp_raw['data'].values,
    })

    coords = cp[['x', 'y']].values
    types = cp['cp_type'].values
    scalars = cp['scalar'].values

    # --- Build edges from separatrices ---
    from collections import defaultdict
    cell_id_to_idx = defaultdict(list)
    for i, cid in enumerate(cp['cell_id']):
        cell_id_to_idx[cid].append(i)

    edges_set = set()
    for _, row in sep_raw.drop_duplicates('SeparatrixId').iterrows():
        src, dst = int(row['SourceId']), int(row['DestinationId'])
        if src in cell_id_to_idx and dst in cell_id_to_idx:
            for si in cell_id_to_idx[src]:
                for di in cell_id_to_idx[dst]:
                    if si != di:
                        edges_set.add((min(si, di), max(si, di)))
    edges = list(edges_set)

    # --- Build hypergraph ---
    hg = build_hypergraph(cp_raw, seg_raw, sep_raw)
    members = {}
    for _, row in hg.iterrows():
        he = eval(row['hyperedge']) if isinstance(row['hyperedge'], str) else row['hyperedge']
        members[row['region_id']] = list(he)

    # --- Compute mu (persistence-based node measure) ---
    mu = compute_mu(scalars, edges, sigma=0.3)

    # --- Compute nu (hyperedge measure) ---
    nu_raw = np.zeros(len(members))
    for i, rid in enumerate(sorted(members.keys())):
        for cp_idx in members[rid]:
            if cp_idx < len(mu):
                nu_raw[i] += mu[cp_idx]
    nu = nu_raw / nu_raw.sum() if nu_raw.sum() > 0 else np.ones(len(nu_raw)) / len(nu_raw)

    # --- Compute omega (hypernetwork function) ---
    diagonal = np.sqrt((coords[:, 0].max() - coords[:, 0].min()) ** 2 +
                       (coords[:, 1].max() - coords[:, 1].min()) ** 2)
    # compute_omega needs (n, 3) coords — add z column
    coords_3d = np.column_stack([coords, cp['z'].values])
    member_lists = [members[rid] for rid in sorted(members.keys())]
    omega, _ = compute_omega(coords_3d, types, edges, member_lists)

    # --- Graph distances for GWD/FGW ---
    adj = build_adjacency_from_edges(len(cp), edges)
    D = compute_graph_distances(adj, coords)

    return {
        'coords': coords, 'types': types, 'scalars': scalars,
        'mu': mu, 'nu': nu, 'omega': omega,
        'D': D, 'diagonal': diagonal,
        'n_cps': len(cp), 'n_regions': len(hg),
    }


def main():
    print("=" * 60)
    print("MS-COOT Demo: Sinusoidal (Clean vs. Noisy)")
    print("=" * 60)

    # Load and prepare both variants
    print("\nPreparing clean surface...")
    clean = load_variant("clean")
    print(f"  CPs: {clean['n_cps']}, Regions: {clean['n_regions']}")

    print("Preparing noisy surface...")
    noisy = load_variant("noisy")
    print(f"  CPs: {noisy['n_cps']}, Regions: {noisy['n_regions']}")

    diagonal = clean['diagonal']

    # --- WD ---
    print("\n--- Wasserstein Distance (WD) ---")
    pi_wd, dist_wd = compute_wd(clean['coords'], noisy['coords'], diagonal)
    tp_wd = compute_type_preservation(pi_wd, clean['types'], noisy['types'])
    print(f"  Distance: {dist_wd:.4f},  Type preservation: {tp_wd:.1%}")

    # --- GWD ---
    print("\n--- Gromov-Wasserstein Distance (GWD) ---")
    D1 = clean['D'] / clean['D'].max() if clean['D'].max() > 0 else clean['D']
    D2 = noisy['D'] / noisy['D'].max() if noisy['D'].max() > 0 else noisy['D']
    pi_gwd, dist_gwd = compute_gwd(D1, D2)
    tp_gwd = compute_type_preservation(pi_gwd, clean['types'], noisy['types'])
    print(f"  Distance: {dist_gwd:.4f},  Type preservation: {tp_gwd:.1%}")

    # --- FGW ---
    print("\n--- Fused Gromov-Wasserstein (FGW) ---")
    M = build_feature_cost(clean['coords'], noisy['coords'], diagonal)
    pi_fgw, dist_fgw = compute_fgw(M, D1, D2, alpha=0.5)
    tp_fgw = compute_type_preservation(pi_fgw, clean['types'], noisy['types'])
    print(f"  Distance: {dist_fgw:.4f},  Type preservation: {tp_fgw:.1%}")

    # --- MS-COOT ---
    print("\n--- MS-COOT (Co-Optimal Transport) ---")
    pi, xi, cost = compute_mscoot(
        clean['omega'], noisy['omega'],
        clean['mu'], noisy['mu'],
        clean['nu'], noisy['nu'],
        types1=clean['types'], types2=noisy['types'],
        alpha=0.5, epsilon=0.001,
    )
    tp_ms = compute_type_preservation(pi, clean['types'], noisy['types'])
    print(f"  Distance: {cost:.4f},  Type preservation: {tp_ms:.1%}")
    print(f"  CP coupling (pi):     {pi.shape[0]} x {pi.shape[1]}")
    print(f"  Region coupling (xi): {xi.shape[0]} x {xi.shape[1]}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"{'Method':<12} {'Distance':>10} {'Type Pres.':>12}")
    print("-" * 36)
    print(f"{'WD':<12} {dist_wd:>10.4f} {tp_wd:>11.1%}")
    print(f"{'GWD':<12} {dist_gwd:>10.4f} {tp_gwd:>11.1%}")
    print(f"{'FGW':<12} {dist_fgw:>10.4f} {tp_fgw:>11.1%}")
    print(f"{'MS-COOT':<12} {cost:>10.4f} {tp_ms:>11.1%}")
    print()
    print("MS-COOT uniquely produces region coupling (xi),")
    print("enabling explicit region-to-region correspondence.")
    print("=" * 60)


if __name__ == "__main__":
    main()
