#!/usr/bin/env python -u
"""COOT parameter sweep for TOSCA ablation study. Run with: python -u scripts/ablation_coot_sweep.py"""
import sys, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # force line buffering

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import ast, warnings
warnings.filterwarnings("ignore")

from ot import coot as ot_coot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mscoot.data_adapter import DatasetAdapter

coot_dir = Path(__file__).resolve().parent.parent / "results/tosca/coot_data"
adapter = DatasetAdapter(Path(__file__).resolve().parent.parent / "datasets/tosca")

labels = np.zeros(80, dtype=int)
CLASS_RANGES = {'cat':(0,11),'centaur':(11,17),'david':(17,24),'dog':(24,33),
                'gorilla':(33,37),'horse':(37,45),'michael':(45,65),'victoria':(65,77),'wolf':(77,80)}
for i, (name, (lo, hi)) in enumerate(CLASS_RANGES.items()):
    labels[lo:hi] = i

def classify(mat, k=1):
    preds = np.zeros(80, dtype=int)
    for i in range(80):
        d = mat[i].copy(); d[i] = np.inf
        nn = np.argsort(d)[:k]
        preds[i] = Counter(labels[nn]).most_common(1)[0][0]
    return (preds == labels).mean()

# Precompute
print("Loading data...", flush=True)
data = {}
for ts in range(80):
    mu_df = pd.read_csv(coot_dir / f"mu_{ts:03d}.csv")
    nu_df = pd.read_csv(coot_dir / f"nu_{ts:03d}.csv")
    vc_df = pd.read_csv(coot_dir / f"vc_{ts:03d}.csv")
    cp = adapter.load_critical_points(ts)
    edges = adapter.load_edges(ts)
    coords = cp[['x','y','z']].values
    n_cp = len(coords)
    n_hyper = len(vc_df)
    members_list = [ast.literal_eval(row['boundary_cps']) for _, row in vc_df.iterrows()]
    centroid_vcs = np.array([coords[m].mean(axis=0) for m in members_list])

    n_total = n_cp + n_hyper
    el = []
    for i, j in edges:
        d = np.linalg.norm(coords[i] - coords[j])
        el.append((i,j,d)); el.append((j,i,d))
    for h, members in enumerate(members_list):
        for ci in members:
            if ci < n_cp:
                d = np.linalg.norm(coords[ci] - centroid_vcs[h])
                el.append((ci,n_cp+h,d)); el.append((n_cp+h,ci,d))
    for i_h in range(n_hyper):
        si = set(members_list[i_h])
        for j_h in range(i_h+1, n_hyper):
            if len(si & set(members_list[j_h])) >= 2:
                d = np.linalg.norm(centroid_vcs[i_h] - centroid_vcs[j_h])
                el.append((n_cp+i_h,n_cp+j_h,d)); el.append((n_cp+j_h,n_cp+i_h,d))
    if el:
        rows,cols,weights = zip(*el)
        adj = csr_matrix((weights,(rows,cols)), shape=(n_total,n_total))
    else:
        adj = csr_matrix((n_total,n_total))
    dm = dijkstra(adj, directed=False, indices=range(n_cp))
    omega = dm[:, n_cp:n_cp+n_hyper]
    fm = np.isfinite(omega) & (omega > 0)
    mf = omega[fm].max() if fm.any() else 1.0
    omega[~np.isfinite(omega)] = mf * 2
    omega = omega / (omega.max() + 1e-8)

    data[ts] = {
        'mu': mu_df['mu'].values, 'nu': nu_df['nu'].values,
        'omega': omega, 'scalars': cp['scalar'].values,
        'types': cp['cp_type'].values,
    }
print("Data loaded.", flush=True)


def run_coot(alpha, epsilon, m_samp_type, omega_norm, nits_bcd):
    mat = np.zeros((80, 80))
    for i in range(80):
        for j in range(i+1, 80):
            d1, d2 = data[i], data[j]
            o1, o2 = d1['omega'].copy(), d2['omega'].copy()
            if omega_norm == 'self':
                o1 /= (o1.max() + 1e-10); o2 /= (o2.max() + 1e-10)
            else:
                s = max(o1.max(), o2.max(), 1e-10); o1 /= s; o2 /= s
            _e = 1e-10
            mu1 = np.maximum(d1['mu'], _e); mu1 /= mu1.sum()
            mu2 = np.maximum(d2['mu'], _e); mu2 /= mu2.sum()
            nu1 = np.maximum(d1['nu'], _e); nu1 /= nu1.sum()
            nu2 = np.maximum(d2['nu'], _e); nu2 /= nu2.sum()

            if m_samp_type == 'scalar':
                M = np.abs(d1['scalars'][:,None] - d2['scalars'][None,:])
            elif m_samp_type == 'type':
                M = (d1['types'][:,None] != d2['types'][None,:]).astype(float)
            elif m_samp_type == 'type+scalar':
                M = (d1['types'][:,None] != d2['types'][None,:]).astype(float) + \
                    np.abs(d1['scalars'][:,None] - d2['scalars'][None,:])

            for eps in ([epsilon, 0] if epsilon > 0 else [0]):
                try:
                    pi, xi, log = ot_coot.co_optimal_transport(
                        X=o1, Y=o2, wx_samp=mu1, wy_samp=mu2,
                        wx_feat=nu1, wy_feat=nu2,
                        epsilon=eps, alpha=alpha, M_samp=M,
                        method_sinkhorn='sinkhorn_log',
                        nits_bcd=nits_bcd, tol_bcd=1e-7,
                        nits_ot=200, tol_sinkhorn=1e-7,
                        log=True, verbose=False)
                    if np.all(np.isfinite(pi)) and np.all(np.isfinite(xi)):
                        break
                except:
                    continue
            o1e = o1[:,np.newaxis,:,np.newaxis]; o2e = o2[np.newaxis,:,np.newaxis,:]
            cost = np.einsum('ijkl,ij,kl->', (o1e-o2e)**2, pi, xi) + alpha * np.sum(M * pi)
            mat[i,j] = mat[j,i] = cost
    return mat


# ─── SWEEP 1: Alpha ───
print("\nSWEEP 1: ALPHA (scalar M_samp, eps=0, self-norm, bcd=100)", flush=True)
print(f"{'alpha':>8} {'k=1':>8} {'k=3':>8} {'k=5':>8}", flush=True)
for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    mat = run_coot(alpha, 0, 'scalar', 'self', 100)
    a1, a3, a5 = classify(mat,1), classify(mat,3), classify(mat,5)
    print(f"{alpha:>8.1f} {a1:>7.1%} {a3:>7.1%} {a5:>7.1%}", flush=True)

# ─── SWEEP 2: Epsilon ───
print("\nSWEEP 2: EPSILON (scalar M_samp, alpha=0.5, self-norm, bcd=100)", flush=True)
print(f"{'epsilon':>10} {'k=1':>8} {'k=3':>8} {'k=5':>8}", flush=True)
for eps in [0, 0.001, 0.01, 0.1]:
    mat = run_coot(0.5, eps, 'scalar', 'self', 100)
    a1, a3, a5 = classify(mat,1), classify(mat,3), classify(mat,5)
    print(f"{eps:>10.4f} {a1:>7.1%} {a3:>7.1%} {a5:>7.1%}", flush=True)

# ─── SWEEP 3: M_samp type ───
print("\nSWEEP 3: M_SAMP TYPE (alpha=0.5, eps=0, self-norm, bcd=100)", flush=True)
print(f"{'M_samp':>15} {'k=1':>8} {'k=3':>8} {'k=5':>8}", flush=True)
for msamp in ['type', 'scalar', 'type+scalar']:
    mat = run_coot(0.5, 0, msamp, 'self', 100)
    a1, a3, a5 = classify(mat,1), classify(mat,3), classify(mat,5)
    print(f"{msamp:>15} {a1:>7.1%} {a3:>7.1%} {a5:>7.1%}", flush=True)

# ─── SWEEP 4: Omega normalization ───
print("\nSWEEP 4: OMEGA NORMALIZATION (scalar M_samp, alpha=0.5, eps=0, bcd=100)", flush=True)
print(f"{'norm':>10} {'k=1':>8} {'k=3':>8} {'k=5':>8}", flush=True)
for norm in ['self', 'joint']:
    mat = run_coot(0.5, 0, 'scalar', norm, 100)
    a1, a3, a5 = classify(mat,1), classify(mat,3), classify(mat,5)
    print(f"{norm:>10} {a1:>7.1%} {a3:>7.1%} {a5:>7.1%}", flush=True)

# ─── SWEEP 5: BCD iterations ───
print("\nSWEEP 5: BCD ITERATIONS (scalar M_samp, alpha=0.5, eps=0, self-norm)", flush=True)
print(f"{'nits_bcd':>10} {'k=1':>8} {'k=3':>8} {'k=5':>8}", flush=True)
for bcd in [10, 50, 100]:
    mat = run_coot(0.5, 0, 'scalar', 'self', bcd)
    a1, a3, a5 = classify(mat,1), classify(mat,3), classify(mat,5)
    print(f"{bcd:>10} {a1:>7.1%} {a3:>7.1%} {a5:>7.1%}", flush=True)

print("\nDone.", flush=True)
