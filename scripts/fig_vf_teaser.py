#!/usr/bin/env python3
"""Viscous Finger teaser — double-column composite figure.

Row 1: (a) Intact 3D MS complex  |  (b) Exploded view with region labels
Row 2: (c) xi coupling (t=51->52) |  (d) MDS ensemble classification

Usage:
    python scripts/fig_vf_teaser.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import rcParams
from matplotlib.image import imread
from matplotlib.patches import Rectangle, Polygon
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.spatial import ConvexHull
from sklearn.manifold import MDS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ── IEEE VIS / TVCG double-column ────────────────────────────────────────────
DOUBLE_COL = 7.16

rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':  'dejavuserif',
    'font.size':         8,
    'axes.titlesize':    8,
    'axes.labelsize':    7,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'legend.fontsize':   6,
    'legend.framealpha': 0.85,
    'legend.edgecolor':  '0.7',
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'axes.linewidth':    0.5,
})

# ── Region color palette (shared with fig_viscous_finger / fig_viscous_finger_xi) ─
PALETTE = [
    (0.217, 0.494, 0.722),  (0.894, 0.102, 0.110),  (0.302, 0.686, 0.290),
    (0.596, 0.306, 0.639),  (1.000, 0.498, 0.000),  (0.200, 0.627, 0.627),
    (0.969, 0.506, 0.749),  (0.800, 0.733, 0.267),  (0.700, 0.150, 0.400),
]
PALETTE_TGT = [PALETTE[0], PALETTE[4], PALETTE[6], PALETTE[5], PALETTE[8]]

# ── Method styling (shared with fig_vf_classification) ────────────────────────
METHOD_ORDER  = ['wd', 'gwd', 'fgw', 'coot']
METHOD_LABELS = {'wd': 'WD', 'gwd': 'GWD', 'fgw': 'FGW', 'coot': 'MS-COOT'}
METHOD_COLORS = {
    'wd': '#377eb8', 'gwd': '#4daf4a', 'fgw': '#ff7f00', 'coot': '#e41a1c',
}

CLASS_ORDER   = ['coarse', 'medium', 'fine']
CLASS_COLORS  = {'coarse': '#fc8d62', 'medium': '#8da0cb', 'fine': '#66c2a5'}
CLASS_MARKERS = {'coarse': 'o',       'medium': 's',       'fine': '^'}
CLASS_LABELS  = {
    'coarse': 'Coarse',
    'medium': 'Medium',
    'fine':   'Fine',
}

DS = 'viscous_finger'
EVENT_T = 51
OUT = Path('results/publication_figures')


# ═══════════════════════════════════════════════════════════════════════════════
# Row 1 helpers (from fig_viscous_finger.py)
# ═══════════════════════════════════════════════════════════════════════════════

def crop_img(img, margin=10):
    if img.ndim == 3 and img.shape[2] >= 3:
        mask = np.any(img[:, :, :3] < 0.95, axis=2)
    else:
        mask = img < 0.95
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows):
        return img, 0, img.shape[0], 0, img.shape[1]
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0 = max(0, r0 - margin); r1 = min(img.shape[0], r1 + margin)
    c0 = max(0, c0 - margin); c1 = min(img.shape[1], c1 + margin)
    return img[r0:r1, c0:c1], r0, r1, c0, c1


def pad_to_same(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    mh, mw = max(h1, h2), max(w1, w2)
    ch = img1.shape[2] if img1.ndim == 3 else 1
    def _pad(im):
        h, w = im.shape[:2]
        out = np.ones((mh, mw, ch), dtype=im.dtype)
        y0, x0 = (mh - h) // 2, (mw - w) // 2
        out[y0:y0 + h, x0:x0 + w] = im
        return out, y0, x0
    p1, dy1, dx1 = _pad(img1)
    p2, dy2, dx2 = _pad(img2)
    return p1, p2, (dy1, dx1, mh, mw), (dy2, dx2, mh, mw)


def show_render(ax, img, ts_label, n_reg=None):
    ax.imshow(img, aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.03, 0.95, f' {ts_label} ',
            transform=ax.transAxes, fontsize=7, fontweight='bold',
            va='top', ha='left', color='white',
            bbox=dict(boxstyle='round,pad=0.10', fc='0.3', ec='none', alpha=0.85))
    if n_reg:
        ax.text(0.97, 0.04, n_reg,
                transform=ax.transAxes, fontsize=6, fontweight='bold',
                va='bottom', ha='right', color='0.2',
                bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='0.4', lw=0.3, alpha=0.85))
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_edgecolor('0.4')


def add_labels(ax, raw_img_shape, crop_bounds, pad_info, json_path, labels, colors, nudges=None):
    orig_h, orig_w = raw_img_shape
    r0, r1, c0, c1 = crop_bounds
    dy, dx, mh, mw = pad_info
    with open(json_path) as f:
        positions = json.load(f)
    if nudges is None:
        nudges = {}
    for rid_str, (jx, jy) in positions.items():
        rid = int(rid_str)
        if rid not in labels:
            continue
        px_x = jx * orig_w
        px_y = jy * orig_h
        crop_x = px_x - c0
        crop_y = px_y - r0
        pad_x = crop_x + dx
        pad_y = crop_y + dy
        ax_x = pad_x / mw
        ax_y = 1.0 - pad_y / mh
        if rid in nudges:
            ax_x += nudges[rid][0]
            ax_y += nudges[rid][1]
        if not (0.01 < ax_x < 0.99 and 0.01 < ax_y < 0.99):
            continue
        ax.text(ax_x, ax_y, labels[rid],
                transform=ax.transAxes, fontsize=7, fontweight='bold',
                ha='center', va='center', color=colors[rid],
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    root = Path(__file__).resolve().parent.parent
    img_dir = root / f'results/{DS}/figures'

    fig = plt.figure(figsize=(DOUBLE_COL, 4.5))

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 1: 3D MS complex renders
    # ─────────────────────────────────────────────────────────────────────────
    gs_row1 = GridSpec(1, 2, figure=fig,
                       left=0.01, right=0.99, top=0.89, bottom=0.52,
                       wspace=0.10)

    # (a) Intact views
    gs_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_row1[0], wspace=0.06)
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])

    img51i, *_ = crop_img(imread(str(img_dir / 'ms3d_051_intact_white.png')))
    img52i, *_ = crop_img(imread(str(img_dir / 'ms3d_052_intact_white.png')))
    img51i, img52i, _, _ = pad_to_same(img51i, img52i)

    show_render(ax_a1, img51i, f't = {EVENT_T}', '9 regions')
    show_render(ax_a2, img52i, f't = {EVENT_T+1}', '5 regions')

    # (b) Exploded views with region labels
    gs_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_row1[1], wspace=0.06)
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[0, 1])

    raw51 = imread(str(img_dir / 'ms3d_051_exploded_white.png'))
    raw52 = imread(str(img_dir / 'ms3d_052_exploded_white.png'))
    crop51, r0_51, r1_51, c0_51, c1_51 = crop_img(raw51, margin=15)
    crop52, r0_52, r1_52, c0_52, c1_52 = crop_img(raw52, margin=15)
    pad51, pad52, pi51, pi52 = pad_to_same(crop51, crop52)

    show_render(ax_b1, pad51, f't = {EVENT_T}', '9 regions')
    show_render(ax_b2, pad52, f't = {EVENT_T+1}', '5 regions')

    labels_51 = {i+1: f'S{i+1}' for i in range(9)}
    labels_52 = {1: 'T1', 2: 'T2', 3: 'T3', 4: 'T4', 5: 'T5'}
    colors_51 = {i+1: PALETTE[i] for i in range(9)}
    colors_52 = {1: PALETTE[0], 2: PALETTE[4], 3: PALETTE[6],
                 4: PALETTE[5], 5: PALETTE[8]}

    nudges_51 = {2: (-0.08, -0.08), 6: (0.0, 0.08), 8: (0.10, 0.14)}
    nudges_52 = {2: (0.04, 0.04)}
    add_labels(ax_b1, raw51.shape[:2], (r0_51, r1_51, c0_51, c1_51), pi51,
               img_dir / 'labels_051.json', labels_51, colors_51, nudges_51)
    add_labels(ax_b2, raw52.shape[:2], (r0_52, r1_52, c0_52, c1_52), pi52,
               img_dir / 'labels_052.json', labels_52, colors_52, nudges_52)

    # Row 1 panel labels
    fig.text(0.01, 0.92, '(a)', fontsize=9, fontweight='bold')
    fig.text(0.04, 0.92, '3D Morse-Smale complex (Viscous Finger)', fontsize=8)
    fig.text(0.51, 0.92, '(b)', fontsize=9, fontweight='bold')
    fig.text(0.54, 0.92, r'Exploded view: 9 regions $\rightarrow$ 5 regions', fontsize=8)

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 2 LEFT: (c) xi coupling matrix
    # ─────────────────────────────────────────────────────────────────────────
    ax_xi = fig.add_axes([0.05, 0.11, 0.35, 0.34])

    consec = pd.read_csv(
        root / f'results/{DS}/metrics/consecutive_metrics.csv')
    xi_raw = pd.read_csv(
        root / f'results/{DS}/couplings/xi_{EVENT_T:03d}_{EVENT_T+1:03d}.csv',
        header=None).values
    rs = xi_raw.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1e-12
    xi_n = xi_raw / rs
    r_ev = consec[consec.ts1 == EVENT_T].iloc[0]
    n_src, n_tgt = int(r_ev.n_regions_1), int(r_ev.n_regions_2)
    if xi_n.shape[0] > n_src:
        xi_n = xi_n[1:]
    xi_n = xi_n[:n_src, :n_tgt]

    ax_xi.imshow(xi_n, cmap='YlOrRd', aspect='auto', vmin=0,
                 vmax=max(0.5, xi_n.max()),
                 interpolation='nearest', origin='upper', rasterized=True)

    for i in range(xi_n.shape[0]):
        for j in range(xi_n.shape[1]):
            val = xi_n[i, j]
            if val >= 0.005:
                color = ('white' if val > 0.5
                         else ('0.15' if val > 0.1 else '0.45'))
                effects = ([pe.withStroke(linewidth=1.2, foreground='0.15')]
                           if val > 0.5 else [])
                ax_xi.text(j, i, f'{val:.0%}', ha='center', va='center',
                           fontsize=7,
                           fontweight='bold' if val > 0.3 else 'normal',
                           color=color, path_effects=effects)

    ax_xi.set_yticks(range(n_src))
    ax_xi.set_yticklabels([])
    ax_xi.set_xticks(range(n_tgt))
    ax_xi.set_xticklabels([])
    ax_xi.tick_params(axis='both', length=0)
    for sp in ax_xi.spines.values():
        sp.set_linewidth(0.4)

    # Color bars — left for source regions (S)
    bar_w = 0.28
    for i in range(n_src):
        bar_x = -0.50 - bar_w
        ax_xi.add_patch(Rectangle((bar_x, i - 0.42), bar_w, 0.84,
                                   fc=PALETTE[i], ec='0.4', lw=0.3,
                                   clip_on=False, zorder=10))
        ax_xi.text(bar_x - 0.08, i, f'S{i+1}',
                   ha='right', va='center', fontsize=7.5, fontweight='bold',
                   color=PALETTE[i], clip_on=False, zorder=15)

    # Color bars — bottom for target regions (T)
    bar_h = 0.28
    for j in range(n_tgt):
        ax_xi.add_patch(Rectangle((j - 0.42, n_src - 0.50), 0.84, bar_h,
                                   fc=PALETTE_TGT[j], ec='0.4', lw=0.3,
                                   clip_on=False, zorder=10))
        ax_xi.text(j, n_src - 0.50 + bar_h + 0.25, f'T{j+1}',
                   ha='center', va='top', fontsize=7.5, fontweight='bold',
                   color=PALETTE_TGT[j], clip_on=False, zorder=15)

    # Highlight split rows: S2 and S9
    for row_idx in [1, 8]:
        ax_xi.add_patch(Rectangle((-0.5, row_idx - 0.45), n_tgt, 0.9,
                                   fc='none', ec='#cc6600', lw=0.8,
                                   ls=(0, (3, 2)), clip_on=True, zorder=11))

    fig.text(0.01, 0.49, '(c)', fontsize=9, fontweight='bold')
    fig.text(0.04, 0.49, r'Region coupling $\xi$ (t=51$\rightarrow$52)', fontsize=8)

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 2 RIGHT: (d) MDS ensemble classification
    # ─────────────────────────────────────────────────────────────────────────
    ens_dir = root / 'results/vf_ensemble'
    meta = pd.read_csv(ens_dir / 'inter_run_distances' / 'metadata.csv')
    mds_matrices = {}
    for m in METHOD_ORDER:
        path = ens_dir / 'inter_run_distances' / f'{m}_matrix.csv'
        if path.exists():
            mds_matrices[m] = pd.read_csv(path, index_col=0).values

    labels_cls = meta['class'].values
    methods = [m for m in METHOD_ORDER if m in mds_matrices]

    mds_left = 0.49
    mds_w   = 0.115
    mds_gap = 0.010
    mds_bot = 0.14
    mds_h   = 0.28

    for col, method in enumerate(methods):
        left = mds_left + col * (mds_w + mds_gap)
        ax = fig.add_axes([left, mds_bot, mds_w, mds_h])
        mat = mds_matrices[method]

        mds = MDS(n_components=2, dissimilarity='precomputed',
                  random_state=42, normalized_stress='auto')
        coords = mds.fit_transform(mat)

        # Convex hulls
        for cn in CLASS_ORDER:
            mask = labels_cls == cn
            pts = coords[mask]
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                poly = Polygon(hull_pts, closed=True,
                               facecolor=CLASS_COLORS[cn], alpha=0.15,
                               edgecolor=CLASS_COLORS[cn], linewidth=0.6,
                               zorder=1)
                ax.add_patch(poly)

        # Scatter
        for cn in CLASS_ORDER:
            mask = labels_cls == cn
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=CLASS_COLORS[cn], marker=CLASS_MARKERS[cn],
                       s=24, edgecolors='k', linewidths=0.3, zorder=3,
                       label=CLASS_LABELS[cn])

        ax.set_xlabel('MDS 1', fontsize=6)
        if col == 0:
            ax.set_ylabel('MDS 2', fontsize=6, labelpad=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)

        # Method title
        ax.set_title(METHOD_LABELS[method], fontsize=7.5, fontweight='bold',
                     color=METHOD_COLORS[method], pad=3)


    fig.text(0.51, 0.49, '(d)', fontsize=9, fontweight='bold')
    fig.text(0.54, 0.49, 'Resolution discrimination (MDS)', fontsize=8)

    # MDS legend — centered below MDS panels
    handles = []
    for cn in CLASS_ORDER:
        h = plt.Line2D([0], [0], marker=CLASS_MARKERS[cn], color='w',
                        markerfacecolor=CLASS_COLORS[cn],
                        markeredgecolor='k', markeredgewidth=0.2,
                        markersize=5, label=CLASS_LABELS[cn], linestyle='')
        handles.append(h)
    fig.text(0.618, 0.08, 'Resolution:', fontsize=6, va='center', ha='right',
             fontstyle='italic')
    fig.legend(handles=handles, loc='center left', ncol=3, frameon=False,
               fontsize=6, bbox_to_anchor=(0.615, 0.08),
               borderpad=0, borderaxespad=0, columnspacing=1.0)

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        out = OUT / f'fig_vf_teaser.{ext}'
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == '__main__':
    main()
