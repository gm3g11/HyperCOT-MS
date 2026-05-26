#!/usr/bin/env python3
"""Heated Cylinder — MS-COOT detects a phase transition at t~56.

Figure (double-column, IEEE VIS / TVCG, 7.16" wide, ~2.8" tall):
  Left  (a): 2×2 distance heatmaps with per-heatmap colorbars
  Right (b): 1×4 MS complex panels with changed-region highlights
"""

import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap, PowerNorm
from matplotlib.colorbar import ColorbarBase
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent

DOUBLE_COL = 7.16
DS = 'heatedcylinder-800-899'
ZOOM_LO, ZOOM_HI = 54, 60

rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.4,
    'xtick.major.width': 0.3, 'ytick.major.width': 0.3,
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'xtick.direction': 'out', 'ytick.direction': 'out',
})

MSTYLE = {
    'WD':      {'color': '#377eb8'},
    'GWD':     {'color': '#4daf4a'},
    'FGW':     {'color': '#ff7f00'},
    'MS-COOT': {'color': '#e41a1c'},
}
METHODS = ['WD', 'GWD', 'FGW', 'MS-COOT']

CP_COLORS = {0: '#377eb8', 1: '#FFFFFF', 2: '#e41a1c'}
CP_EDGES  = {0: '#1a4a7a', 1: '#444444', 2: '#a01015'}
CP_LABELS = {0: 'min', 1: 'saddle', 2: 'max'}

MS_XLIM = (-0.55, 0.55)
MS_YLIM = (-0.35, 1.65)  # slight bottom crop, top cropped above highest CP (~1.58)

_SCALAR_CMAP = LinearSegmentedColormap.from_list(
    'scalar_field', plt.get_cmap('Oranges')(np.linspace(0.05, 0.80, 256)))

HIGHLIGHT_COLOR = '#2ca02c'  # green


def load_matrix(path):
    with open(path) as f:
        c = f.read(1)
    return (pd.read_csv(path, index_col=0).values if c == ','
            else np.loadtxt(path, delimiter=','))


def load_all_matrices():
    dm = ROOT / f"results/{DS}/distance_matrices"
    return {name: load_matrix(dm / fn) for name, fn in
            [('WD', 'wd_matrix.csv'), ('GWD', 'gwd_matrix.csv'),
             ('FGW', 'fgw_matrix.csv'), ('MS-COOT', 'coot_matrix.csv')]}


def build_grids(seg_ts, min_region_px=15):
    x, y = seg_ts['Points:0'].values, seg_ts['Points:1'].values
    xu = np.sort(np.unique(np.round(x, 6)))
    yu = np.sort(np.unique(np.round(y, 6)))
    dx, dy = np.median(np.diff(xu)), np.median(np.diff(yu))
    nx, ny = len(xu), len(yu)
    xi = np.clip(np.round((x - xu[0]) / dx).astype(int), 0, nx - 1)
    yi = np.clip(np.round((y - yu[0]) / dy).astype(int), 0, ny - 1)
    scalar_grid = np.full((ny, nx), np.nan)
    region_grid = np.full((ny, nx), -1, dtype=int)
    scalar_grid[yi, xi] = seg_ts['Scalars_'].values
    region_grid[yi, xi] = seg_ts['MorseSmaleManifold'].values.astype(int)
    ext = [xu[0] - dx/2, xu[-1] + dx/2, yu[0] - dy/2, yu[-1] + dy/2]
    if min_region_px > 0:
        unique, counts = np.unique(region_grid[region_grid >= 0],
                                   return_counts=True)
        tiny = set(unique[counts < min_region_px])
        for rid in tiny:
            mask = region_grid == rid
            ys_, xs_ = np.where(mask)
            nc = {}
            for py, px in zip(ys_, xs_):
                for ddy, ddx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny_, nx_ = py + ddy, px + ddx
                    if 0 <= ny_ < ny and 0 <= nx_ < nx:
                        nid = region_grid[ny_, nx_]
                        if nid >= 0 and nid != rid and nid not in tiny:
                            nc[nid] = nc.get(nid, 0) + 1
            if nc:
                region_grid[mask] = max(nc, key=nc.get)
    return scalar_grid, region_grid, ext


def find_changed_regions(rg_src, rg_tgt):
    regs_t = np.unique(rg_tgt[rg_tgt >= 0])
    tgt_best_src, tgt_best_count = {}, {}
    for rid_t in regs_t:
        mask_t = rg_tgt == rid_t
        src_ids = rg_src[mask_t]
        src_ids = src_ids[src_ids >= 0]
        if len(src_ids) == 0:
            tgt_best_src[rid_t] = -1
            tgt_best_count[rid_t] = 0
            continue
        u, c = np.unique(src_ids, return_counts=True)
        tgt_best_src[rid_t] = u[c.argmax()]
        tgt_best_count[rid_t] = c.max()
    src_to_tgts = defaultdict(list)
    for rid_t in regs_t:
        src_to_tgts[tgt_best_src[rid_t]].append(rid_t)
    changed = set()
    for src_rid, tgt_rids in src_to_tgts.items():
        if src_rid == -1:
            changed.update(tgt_rids)
        elif len(tgt_rids) > 1:
            s = sorted(tgt_rids, key=lambda r: tgt_best_count[r], reverse=True)
            changed.update(s[1:])
    return changed


def _build_polylines(sep_df):
    """Build polylines from TTK separatrix geometry CSV."""
    lines = []
    for sid in sep_df['SeparatrixId'].unique():
        seg = sep_df[sep_df.SeparatrixId == sid].sort_values('PointOrder')
        pts = seg[['Points:0', 'Points:1']].values
        if len(pts) >= 2:
            lines.append(pts)
    return lines


def render_ms(ax, scalar_grid, region_grid, ext, cp_ts,
              sc_vmin, sc_vmax, changed_rids=None,
              sep_polylines=None, cp_size=14):
    norm = PowerNorm(gamma=0.27, vmin=sc_vmin, vmax=sc_vmax)
    ax.imshow(scalar_grid, cmap=_SCALAR_CMAP, origin='lower', aspect='equal',
              norm=norm, interpolation='bilinear', extent=ext, rasterized=True)
    if changed_rids is None:
        changed_rids = set()

    # TTK separatrices
    if sep_polylines is not None:
        for pl in sep_polylines:
            ax.plot(pl[:, 0], pl[:, 1], 'k-', lw=0.4, zorder=2,
                    solid_capstyle='round', clip_on=True)

    # Highlight changed regions with bold contour
    if changed_rids:
        ny, nx = region_grid.shape
        pad = 2
        dx = (ext[1] - ext[0]) / nx
        dy = (ext[3] - ext[2]) / ny
        ext_pad = [ext[0] - pad*dx, ext[1] + pad*dx,
                   ext[2] - pad*dy, ext[3] + pad*dy]
        for rid in changed_rids:
            raw = (region_grid == rid).astype(float)
            padded = np.pad(raw, pad, mode='constant', constant_values=0)
            m = gaussian_filter(padded, sigma=0.5)
            ax.contour(m, levels=[0.5], colors=[HIGHLIGHT_COLOR],
                       linewidths=[1.5], linestyles='solid',
                       origin='lower', extent=ext_pad, zorder=3)

    # CPs — min/max first, saddles on top
    for cp_type in [0, 2, 1]:
        mask = cp_ts['CellDimension'] == cp_type
        if not mask.any():
            continue
        z = 6 if cp_type == 1 else 5
        ax.scatter(cp_ts.loc[mask, 'Points:0'], cp_ts.loc[mask, 'Points:1'],
                   c=CP_COLORS[cp_type], marker='o', s=cp_size,
                   edgecolors=CP_EDGES[cp_type], linewidths=0.8, zorder=z)
    ax.set_xlim(MS_XLIM)
    ax.set_ylim(MS_YLIM)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    print('Loading data...')
    matrices = load_all_matrices()
    nt = matrices['WD'].shape[0]
    seg_all = pd.read_csv(ROOT / f'datasets/{DS}/ttk_output/segmentation.csv')
    cp_all = pd.read_csv(ROOT / f'datasets/{DS}/ttk_output/critical_points.csv')

    TS_LIST = [55, 56, 57, 58]
    grids = {t: build_grids(seg_all[seg_all['TimeStep'] == t]) for t in TS_LIST}
    sc_vmin = min(np.nanmin(grids[t][0]) for t in TS_LIST)
    sc_vmax = max(np.nanmax(grids[t][0]) for t in TS_LIST)

    changed_per_t = {TS_LIST[0]: set()}
    reg_counts = {t: len(np.unique(grids[t][1][grids[t][1] >= 0])) for t in TS_LIST}
    reg_counts[55] = 30  # manual override: TTK raw count for this frame
    for t_s, t_t in zip(TS_LIST[:-1], TS_LIST[1:]):
        changed_per_t[t_t] = find_changed_regions(grids[t_s][1], grids[t_t][1])

    # ── Figure ──
    print('Building figure...')
    fig = plt.figure(figsize=(DOUBLE_COL, 3.2), facecolor='white')

    A_L, A_R = 0.04, 0.34
    B_L, B_R = 0.40, 0.995

    # (a) left: 2×2 heatmaps
    gs_left = GridSpec(2, 2, figure=fig,
                       left=A_L, right=A_R,
                       top=0.93, bottom=0.30,
                       wspace=0.40, hspace=0.01)

    # (b) right: 1×4 MS panels
    gs_ms = GridSpec(1, 4, figure=fig,
                     left=B_L, right=B_R,
                     top=0.94, bottom=0.20,
                     wspace=0.05)

    # ── (a) 2×2 heatmaps ──
    hm_axes = []
    vmax_per = {}
    hm_pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (r, c), name in zip(hm_pos, METHODS):
        mat = matrices[name]
        col = MSTYLE[name]['color']
        ax = fig.add_subplot(gs_left[r, c])
        hm_axes.append(ax)
        pos = mat[~np.eye(nt, dtype=bool)]
        pos = pos[pos > 0]
        vmax = np.percentile(pos, 97) if len(pos) else 1.0
        vmax_per[name] = vmax
        ax.imshow(mat, cmap='viridis', aspect='equal', vmin=0, vmax=vmax,
                  interpolation='nearest', origin='upper', rasterized=True)
        ax.set_title(name, fontsize=8, fontweight='bold', color=col, pad=2)
        ax.set_yticks([0, 50, nt - 1] if c == 0 else [])
        ax.set_xticks([0, 50, nt - 1] if r == 1 else [])
        ax.tick_params(labelsize=7)
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
        ax.set_box_aspect(1)
        mid = (ZOOM_LO + ZOOM_HI) / 2
        ax.annotate('', xy=(mid, mid),
                    xytext=(mid + 15, mid - 15),
                    arrowprops=dict(arrowstyle='->', color='#ff3333',
                                   lw=1.5, shrinkA=0, shrinkB=2),
                    zorder=8)

    # y-labels
    hm_axes[0].set_ylabel('Timestep ($t$)', fontsize=8, labelpad=1)
    hm_axes[2].set_ylabel('Timestep ($t$)', fontsize=8, labelpad=1)

    # Per-heatmap colorbars (ionfront style)
    fig.canvas.draw()
    for idx, name in enumerate(METHODS):
        vmax = vmax_per[name]
        pos = hm_axes[idx].get_position()
        cb_ax = fig.add_axes([pos.x1 + 0.008, pos.y0 + pos.height * 0.15,
                              0.004, pos.height * 0.55])
        cb = ColorbarBase(cb_ax, cmap=plt.get_cmap('viridis'),
                          norm=Normalize(0, vmax), orientation='vertical')
        if vmax > 0:
            exp = int(np.floor(np.log10(vmax)))
            coeff = vmax / 10**exp
            cb.set_ticks([0, vmax])
            cb.set_ticklabels(['0', f'{coeff:.1f}'])
            cb_pos2 = cb.ax.get_position()
            fig.text(cb_pos2.x1 + 0.001, cb_pos2.y1 + 0.012,
                     r'$\times 10^{' + str(exp) + r'}$',
                     fontsize=7, va='bottom', ha='left')
        else:
            cb.set_ticks([0])
            cb.set_ticklabels(['0'])
        cb.ax.tick_params(labelsize=7, width=0.2, length=1.0, pad=1)
        cb.outline.set_linewidth(0.2)
        # "Distance" label on GWD and MS-COOT
        if name in ('GWD', 'MS-COOT'):
            cb_pos3 = cb.ax.get_position()
            fig.text(cb_pos3.x1 + 0.012, cb_pos3.y0 + cb_pos3.height / 2,
                     'Distance', fontsize=7, rotation=90,
                     ha='left', va='center', color='black', fontstyle='italic')

    # (a) title — ionfront style: bold (a) + regular text
    fig.text(0.04, 0.96, '(a)', fontsize=10, fontweight='bold')
    fig.text(0.07, 0.96, 'Distance matrices \u2014 Heated Cylinder', fontsize=9)

    # ── (b) MS complex panels ──
    print('Rendering MS complex panels...')

    # Load TTK separatrix geometry and reconstruct missing CPs
    ttk_dir = ROOT / f'datasets/{DS}/ttk_output'
    sep_polylines = {}
    for t in TS_LIST:
        sep_path = ttk_dir / f'separatrix_geometry_{t}.csv'
        sep_df = pd.read_csv(sep_path)
        sep_polylines[t] = _build_polylines(sep_df)
        # Reconstruct missing CPs
        cp_cellids = set(cp_all[cp_all.TimeStep == t]['CellId'].astype(int))
        new_rows = []
        seen = set()
        for sid in sep_df['SeparatrixId'].unique():
            seg = sep_df[sep_df.SeparatrixId == sid].sort_values('PointOrder')
            pts = seg[['Points:0', 'Points:1']].values
            src = int(seg.iloc[0]['SourceId'])
            dst = int(seg.iloc[0]['DestinationId'])
            stype = int(seg.iloc[0]['SeparatrixType'])
            if src not in cp_cellids and src not in seen:
                new_rows.append({'TimeStep': t, 'CellId': src, 'CellDimension': 1,
                                 'Points:0': pts[0, 0], 'Points:1': pts[0, 1]})
                seen.add(src)
            if dst not in cp_cellids and dst not in seen:
                dim = 0 if stype == 0 else 2
                new_rows.append({'TimeStep': t, 'CellId': dst, 'CellDimension': dim,
                                 'Points:0': pts[-1, 0], 'Points:1': pts[-1, 1]})
                seen.add(dst)
        if new_rows:
            cp_all = pd.concat([cp_all, pd.DataFrame(new_rows)], ignore_index=True)
            print(f'  Reconstructed {len(new_rows)} missing CPs for t={t}')

    ms_axes = []
    for col_i, t in enumerate(TS_LIST):
        sg, rg, ext = grids[t]
        cp_ts = cp_all[cp_all['TimeStep'] == t].copy()
        n_reg = reg_counts[t]

        ax = fig.add_subplot(gs_ms[0, col_i])
        ms_axes.append(ax)
        render_ms(ax, sg, rg, ext, cp_ts, sc_vmin, sc_vmax,
                  changed_rids=changed_per_t[t],
                  sep_polylines=sep_polylines[t], cp_size=12)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f'$t = {t}$  ({n_reg} regions)',
                     fontsize=7, fontweight='bold', pad=1)

    # (b) title
    fig.canvas.draw()
    fig.text(B_L, 0.96, '(b)', fontsize=10, fontweight='bold')
    fig.text(B_L + 0.03, 0.96, 'MS complex', fontsize=9)

    # Arrows between consecutive MS panels
    for i in range(len(ms_axes) - 1):
        bb_l = ms_axes[i].get_position()
        bb_r = ms_axes[i + 1].get_position()
        arrow_x = (bb_l.x1 + bb_r.x0) / 2
        arrow_y = (bb_l.y0 + bb_l.y1) / 2
        fig.text(arrow_x, arrow_y, r'$\rightarrow$',
                 fontsize=10, ha='center', va='center',
                 color='0.3')

    # CP legend + scalar colorbar — on title line
    legend_elements = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=CP_COLORS[k], markeredgecolor=CP_EDGES[k],
               markeredgewidth=0.8,
               markersize=5, label=CP_LABELS[k])
        for k in [0, 1, 2]
    ]

    leg_ax = fig.add_axes([B_L + 0.14, 0.96 - 0.026, 0.22, 0.018])
    leg_ax.set_axis_off()
    leg_ax.legend(handles=legend_elements, loc='center left',
                  fontsize=7, ncol=3, framealpha=0.0,
                  borderpad=0, handletextpad=0.2, columnspacing=0.5)

    # Scalar colorbar
    cbar_x0 = 0.80
    cbar_w = 0.13
    cbar_ax = fig.add_axes([cbar_x0, 0.96 - 0.021, cbar_w, 0.008])
    cb = ColorbarBase(cbar_ax, cmap=_SCALAR_CMAP,
                      norm=Normalize(0, 1), orientation='horizontal')
    cb.set_ticks([])
    cb.outline.set_linewidth(0.2)
    fig.text(cbar_x0 - 0.005, 0.96 - 0.017, 'Low',
             fontsize=7, ha='right', va='center')
    fig.text(cbar_x0 + cbar_w + 0.005, 0.96 - 0.017, 'High',
             fontsize=7, ha='left', va='center')
    fig.text(cbar_x0 + cbar_w / 2, 0.96 + 0.004, 'Velocity',
             fontsize=7, ha='center', va='center', fontstyle='italic')

    # Save
    out = ROOT / "results/publication_figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        p = out / f'fig_heatedcylinder_phase.{ext}'
        fig.savefig(p, facecolor='white')
        print(f'  Saved {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()
