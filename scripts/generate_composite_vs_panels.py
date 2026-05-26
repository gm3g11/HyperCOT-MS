#!/usr/bin/env python3
"""Composite figure: Vortex Street heatmaps + zoom + MS complex panels.

Layout (single-column, IEEE VIS / TVCG, 3.5" wide):
  Top row:  (a) 2x2 heatmaps | (b) 2x2 zoom  — side by side, aligned
  Shared magma colorbar between top and bottom
  Bottom:   (c) t=108 and t=109 MS complex panels stacked
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from matplotlib.image import imread
from matplotlib.lines import Line2D
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SINGLE_COL = 3.5
DOUBLE_COL = 7.0

rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.4,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.major.size': 2.0,
    'ytick.major.size': 2.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

MSTYLE = {
    'WD':       {'color': '#377eb8'},
    'GWD':      {'color': '#4daf4a'},
    'FGW':      {'color': '#ff7f00'},
    'MS-COOT':  {'color': '#e41a1c'},
}

CP_COLORS_FIG = {0: '#377eb8', 1: '#FFFFFF', 2: '#e41a1c'}
CP_EDGES_FIG = {0: '#1a4a7a', 1: '#444444', 2: '#a01015'}
CP_LABELS_FIG = {0: 'min', 1: 'saddle', 2: 'max'}

# Scalar field colormap — Oranges, wider range than original
_SCALAR_CMAP = LinearSegmentedColormap.from_list(
    'scalar_field', plt.get_cmap('Oranges')(np.linspace(0.05, 0.80, 256)))

EV_TS = (108, 109)
EV_COLOR = '#e41a1c'
ZOOM_RANGE = (107, 111)
ZOOM_EC = '#e41a1c'
WIDE_XLIM = (-0.5, 199.5)   # show left half of domain
ZOOM_XY = (17, 48, 17, 32)  # left: cover corner, right: include saddle at (45.5,26.5)


def _build_grids(seg_ts, min_region_px=15):
    """Build 2D scalar and region grids from segmentation dataframe.

    Tiny regions (< min_region_px pixels) are merged into their largest
    4-connected neighbor to avoid spurious boundary artifacts.
    """
    x = seg_ts['Points:0'].astype(int).values
    y = seg_ts['Points:1'].astype(int).values
    nx, ny = x.max() + 1, y.max() + 1
    scalar_grid = np.full((ny, nx), np.nan)
    region_grid = np.full((ny, nx), -1, dtype=int)
    scalar_grid[y, x] = seg_ts['Scalars_'].values
    region_grid[y, x] = seg_ts['MorseSmaleManifold'].values

    # Merge tiny regions into largest neighbor
    if min_region_px > 0:
        unique, counts = np.unique(region_grid[region_grid >= 0],
                                   return_counts=True)
        tiny = set(unique[counts < min_region_px])
        if tiny:
            # Count neighboring region pixels for each tiny region
            for rid in tiny:
                mask = region_grid == rid
                ys, xs = np.where(mask)
                neighbor_counts = {}
                for py, px in zip(ys, xs):
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny_, nx_ = py + dy, px + dx
                        if 0 <= ny_ < ny and 0 <= nx_ < nx:
                            nid = region_grid[ny_, nx_]
                            if nid >= 0 and nid != rid and nid not in tiny:
                                neighbor_counts[nid] = neighbor_counts.get(nid, 0) + 1
                if neighbor_counts:
                    best = max(neighbor_counts, key=neighbor_counts.get)
                    region_grid[mask] = best

    return scalar_grid, region_grid, nx, ny


def _boundary_mask(region_grid):
    """Extract MS boundary pixels via 8-connectivity."""
    ny, nx = region_grid.shape
    bnd = np.zeros((ny, nx), dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        shifted = np.roll(np.roll(region_grid, -dy, axis=0), -dx, axis=1)
        bnd |= (region_grid != shifted)
    # Mark domain edges as boundary
    bnd[0, :] = bnd[-1, :] = bnd[:, 0] = bnd[:, -1] = True
    return bnd


def _extract_edges(cp_ts, sep_ts):
    """Extract graph edges as (src_xy, dst_xy) from separatrices."""
    # Build CellId → position map (handle duplicate CellIds via CellDimension)
    cellid_dim_to_xy = {}
    for _, row in cp_ts.iterrows():
        key = (int(row['CellId']), int(row['CellDimension']))
        cellid_dim_to_xy[key] = (row['Points:0'], row['Points:1'])

    edges_df = sep_ts.groupby('SeparatrixId').first()[
        ['SourceId', 'DestinationId', 'SeparatrixType']]

    edges = []
    for _, e in edges_df.iterrows():
        src_key = (int(e['SourceId']), 1)  # source = saddle
        dst_dim = 0 if int(e['SeparatrixType']) == 0 else 2
        dst_key = (int(e['DestinationId']), dst_dim)
        if src_key in cellid_dim_to_xy and dst_key in cellid_dim_to_xy:
            edges.append((cellid_dim_to_xy[src_key],
                          cellid_dim_to_xy[dst_key]))
    return edges


def _build_polylines(sep_geom):
    """Build polylines from TTK separatrix geometry CSV."""
    lines = []
    for sid in sep_geom['SeparatrixId'].unique():
        seg = sep_geom[sep_geom.SeparatrixId == sid].sort_values('PointOrder')
        pts = seg[['Points:0', 'Points:1']].values
        if len(pts) >= 2:
            lines.append(pts)
    return lines


def _render_ms_on_ax(ax, scalar_grid, region_grid, cp_ts, cmap,
                     vmin=None, vmax=None, cp_size=12,
                     bnd_lw=0.5, sep_polylines=None,
                     cp_edge_lw=0.3, xlim=None, ylim=None,
                     aspect='equal'):
    """Render scalar field + TTK separatrices + CPs."""
    ny, nx = scalar_grid.shape
    if vmin is None:
        vmin = np.nanmin(scalar_grid)
    if vmax is None:
        vmax = np.nanmax(scalar_grid)

    # Layer 0: Scalar field background
    ax.imshow(scalar_grid, cmap=cmap, origin='lower', aspect='equal',
              vmin=vmin, vmax=vmax, interpolation='gaussian', rasterized=True)

    # Layer 1: TTK separatrices
    if sep_polylines is not None:
        for pl in sep_polylines:
            ax.plot(pl[:, 0], pl[:, 1], 'k-', lw=bnd_lw, zorder=2,
                    solid_capstyle='round')



    # Layer 2: CPs — min/max first, saddles on top for visibility
    for cp_type in [0, 2, 1]:
        mask = cp_ts['CellDimension'] == cp_type
        if not mask.any():
            continue
        z = 6 if cp_type == 1 else 5  # saddles above min/max
        ax.scatter(cp_ts.loc[mask, 'Points:0'], cp_ts.loc[mask, 'Points:1'],
                   c=CP_COLORS_FIG[cp_type], marker='o', s=cp_size,
                   edgecolors=CP_EDGES_FIG[cp_type], linewidths=0.8, zorder=z)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])


def load_matrix(path):
    with open(path) as f:
        c = f.read(1)
    return pd.read_csv(path, index_col=0).values if c == ',' else np.loadtxt(path, delimiter=',')


def load_all():
    root = Path(__file__).resolve().parent.parent
    dm = root / "results/publication_figures/crc/results/vortexstreet/distance_matrices"
    return {name: load_matrix(dm / fn) for name, fn in
            [('WD', 'wd_matrix.csv'), ('GWD', 'gwd_matrix.csv'),
             ('FGW', 'fgw_matrix.csv'), ('MS-COOT', 'coot_matrix.csv')]}


# -- Heatmap (full) --------------------------------------------------------

def draw_heatmap(ax, mat, name):
    nt = mat.shape[0]
    off = mat[~np.eye(nt, dtype=bool)]
    pos = off[off > 0]
    vmax = np.percentile(pos, 97) if len(pos) else 1.0
    is_hc = name == 'MS-COOT'

    if is_hc:
        med = np.median(pos)
        r = med / vmax if vmax > 0 else 1.0
        gamma = np.clip(np.log(0.5) / np.log(r), 0.3, 3.0) if 0 < r < 1 else 1.0
        norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
        ax.imshow(mat, cmap='viridis', aspect='equal', norm=norm,
                  interpolation='nearest', origin='upper', rasterized=True)
    else:
        norm = None
        ax.imshow(mat, cmap='viridis', aspect='equal', vmin=0, vmax=vmax,
                  interpolation='nearest', origin='upper', rasterized=True)

    step = 50
    ax.set_xticks(list(range(0, nt, step)))
    ax.set_yticks(list(range(0, nt, step)))
    ax.tick_params(labelsize=6)
    ax.set_box_aspect(1)  # force square axes → data fills exactly, no "window" gap
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)

    # Arrow pointing to zoom region
    r0, r1 = ZOOM_RANGE
    mid = (r0 + r1) / 2
    ax.annotate('', xy=(mid, mid), xytext=(mid + 25, mid - 25),
                arrowprops=dict(arrowstyle='->', color=ZOOM_EC, lw=1.2),
                zorder=4)
    return vmax, norm


# -- Heatmap (zoomed) ------------------------------------------------------

def draw_heatmap_zoom(ax, mat, name, vmax, norm_obj=None):
    r0, r1 = ZOOM_RANGE
    n = r1 - r0
    zmat = mat[r0:r1, r0:r1]
    is_hc = name == 'MS-COOT'

    if is_hc and norm_obj is not None:
        ax.imshow(zmat, cmap='viridis', aspect='equal', norm=norm_obj,
                  interpolation='nearest', origin='upper', rasterized=True)
    else:
        ax.imshow(zmat, cmap='viridis', aspect='equal', vmin=0, vmax=vmax,
                  interpolation='nearest', origin='upper', rasterized=True)

    key_ts = list(range(r0, r1))
    show_ts = [t for t in key_ts if t % 2 == 0]  # show only even: 108, 110
    zt_all = [t - r0 for t in key_ts]
    zt_show = [t - r0 for t in show_ts]
    ax.set_xticks(zt_show)
    ax.set_yticks(zt_show)
    ax.set_xticklabels([str(t) for t in show_ts], fontsize=7)
    ax.set_yticklabels([str(t) for t in show_ts], fontsize=7)

    ax.set_box_aspect(1)  # force square axes
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)

    for k in range(1, n):
        ax.axhline(k - 0.5, color='white', lw=0.12, alpha=0.3, zorder=2)
        ax.axvline(k - 0.5, color='white', lw=0.12, alpha=0.3, zorder=2)

    t1, t2 = EV_TS
    zt1, zt2 = t1 - r0, t2 - r0
    if 0 <= zt1 < n and 0 <= zt2 < n:
        ax.add_patch(Rectangle((zt2 - 0.5, zt1 - 0.5), 1, 1,
                                lw=1.5, ec=EV_COLOR, fc='none', zorder=6))


# -- Main ------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent.parent
    matrices = load_all()

    # ── Load TTK data for programmatic MS rendering ──
    ttk_dir = root / "datasets/vortexstreet/ttk_output"
    seg_all = pd.read_csv(ttk_dir / "segmentation.csv")
    cp_all = pd.read_csv(ttk_dir / "critical_points.csv")
    # Load TTK separatrix geometry and reconstruct missing CPs from endpoints
    sep_geom_raw = {}
    for ts in [108, 109]:
        sep_df = pd.read_csv(ttk_dir / f"separatrix_geometry_{ts}.csv")
        sep_geom_raw[ts] = sep_df
    sep_geom = {ts: _build_polylines(sep_geom_raw[ts]) for ts in [108, 109]}

    # Reconstruct CPs that separatrices reference but are missing from CP CSV
    for ts in [108, 109]:
        sep_df = sep_geom_raw[ts]
        cp_ts = cp_all[cp_all.TimeStep == ts]
        cp_cellids = set(cp_ts['CellId'].astype(int).values)
        new_rows = []
        seen = set()
        for sid in sep_df['SeparatrixId'].unique():
            seg = sep_df[sep_df.SeparatrixId == sid].sort_values('PointOrder')
            pts = seg[['Points:0', 'Points:1']].values
            src = int(seg.iloc[0]['SourceId'])
            dst = int(seg.iloc[0]['DestinationId'])
            stype = int(seg.iloc[0]['SeparatrixType'])
            # Source is always a saddle (CellDimension=1)
            if src not in cp_cellids and src not in seen:
                new_rows.append({
                    'TimeStep': ts, 'CellId': src, 'CellDimension': 1,
                    'Points:0': pts[0, 0], 'Points:1': pts[0, 1], 'Points:2': 0.0,
                })
                seen.add(src)
            # Destination: min (dim=0) for type 0, max (dim=2) for type 1
            if dst not in cp_cellids and dst not in seen:
                dim = 0 if stype == 0 else 2
                new_rows.append({
                    'TimeStep': ts, 'CellId': dst, 'CellDimension': dim,
                    'Points:0': pts[-1, 0], 'Points:1': pts[-1, 1], 'Points:2': 0.0,
                })
                seen.add(dst)
        if new_rows:
            cp_all = pd.concat([cp_all, pd.DataFrame(new_rows)], ignore_index=True)
            print(f"  Reconstructed {len(new_rows)} missing CPs for t={ts}")

    # ── Layout: Double column, 3 rows ──
    fig = plt.figure(figsize=(DOUBLE_COL, 5.5))

    # Row 1: (a) 1×4 heatmaps + (b) 2×2 zoom
    gs_a = GridSpec(1, 4, figure=fig,
                    left=0.05, right=0.74,
                    top=0.94, bottom=0.68,
                    wspace=0.30)

    gs_b = GridSpec(2, 2, figure=fig,
                    left=0.78, right=0.99,
                    top=0.92, bottom=0.70,
                    wspace=0.01, hspace=0.39)

    # Row 2+3: (c) t=108 and t=109 — use a single 2×2 GridSpec
    gs_c = GridSpec(2, 2, figure=fig,
                    left=0.01, right=0.99,
                    top=0.72, bottom=0.15,
                    wspace=0.02, hspace=-0.4,
                    width_ratios=[2.0, 1])

    # ── (a) 1×4 Heatmaps ──
    methods = ['WD', 'GWD', 'FGW', 'MS-COOT']
    vmax_c, norm_c = {}, {}
    hm_axes = []

    for idx, name in enumerate(methods):
        ax = fig.add_subplot(gs_a[0, idx])
        hm_axes.append(ax)
        vmax_c[name], norm_c[name] = draw_heatmap(ax, matrices[name], name)

    # Only leftmost heatmap keeps y-labels
    for ax in hm_axes[1:]:
        ax.set_yticklabels([])

    # "Timestep (t)" on y-axis of WD only
    fig.canvas.draw()
    bb = hm_axes[0].get_position()
    fig.text(bb.x0 - 0.035, bb.y0 + bb.height / 2,
             r'Timestep ($t$)', fontsize=8, ha='center', va='center', rotation=90)

    # ── (b) 2×2 Zoom ──
    zm_axes = []

    for idx, name in enumerate(methods):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(gs_b[r, c])
        zm_axes.append(ax)
        draw_heatmap_zoom(ax, matrices[name], name,
                          vmax=vmax_c[name], norm_obj=norm_c[name])

    # Top row: hide x-labels. All: hide y-labels
    for ax in zm_axes[:2]:
        ax.set_xticklabels([])
    for ax in zm_axes:
        ax.set_yticklabels([])

    # ── (c) MS complex — t=108 and t=109 side by side ──
    first_ms_ax = None

    # Compute shared scalar range across both timesteps
    s108 = seg_all[seg_all.TimeStep == 108]['Scalars_']
    s109 = seg_all[seg_all.TimeStep == 109]['Scalars_']
    sc_vmin = min(s108.min(), s109.min())
    sc_vmax = max(s108.max(), s109.max())

    zx0, zx1, zy0, zy1 = ZOOM_XY

    for i, ts in enumerate([108, 109]):
        seg_ts = seg_all[seg_all.TimeStep == ts]
        cp_ts = cp_all[cp_all.TimeStep == ts].copy()
        scalar_grid, region_grid, nx, ny = _build_grids(seg_ts)

        # Wide view
        ax_w = fig.add_subplot(gs_c[i, 0])
        if i == 0:
            first_ms_ax = ax_w
        _render_ms_on_ax(ax_w, scalar_grid, region_grid, cp_ts,
                         cmap=_SCALAR_CMAP, vmin=sc_vmin, vmax=sc_vmax,
                         cp_size=20, bnd_lw=0.4,
                         sep_polylines=sep_geom[ts],
                         cp_edge_lw=0.4, xlim=WIDE_XLIM)
        for sp in ax_w.spines.values():
            sp.set_linewidth(0.3)
            sp.set_edgecolor('0.4')

        # Timestamp label
        ax_w.text(0.01, 0.90, f't = {ts}', transform=ax_w.transAxes,
                  fontsize=8, fontweight='bold', va='top',
                  bbox=dict(facecolor='white', edgecolor='0.5',
                            lw=0.3, pad=1.5, alpha=0.8))

        # Zoom box on wide view
        ax_w.add_patch(Rectangle((zx0 - 0.5, zy0 - 0.5), zx1 - zx0, zy1 - zy0,
                                 lw=1.2, ec=ZOOM_EC, fc='none', zorder=6))

        # Zoom view
        ax_z = fig.add_subplot(gs_c[i, 1])
        _render_ms_on_ax(ax_z, scalar_grid, region_grid, cp_ts,
                         cmap=_SCALAR_CMAP, vmin=sc_vmin, vmax=sc_vmax,
                         cp_size=45, bnd_lw=0.7,
                         sep_polylines=sep_geom[ts],
                         cp_edge_lw=0.5,
                         xlim=(zx0 - 0.5, zx1 + 0.5),
                         ylim=(zy0 - 0.5, zy1 + 0.5))
        for sp in ax_z.spines.values():
            sp.set_linewidth(1.2)
            sp.set_edgecolor(ZOOM_EC)

    # ── Labels & annotations ──
    fig.canvas.draw()

    # Method names above each heatmap
    for ax, name in zip(hm_axes, methods):
        bb = ax.get_position()
        fig.text(bb.x0 + bb.width / 2, bb.y1 + 0.005, name,
                 fontsize=8, fontweight='bold', color=MSTYLE[name]['color'],
                 ha='center', va='bottom')
    for ax, name in zip(zm_axes, methods):
        bb = ax.get_position()
        fig.text(bb.x0 + bb.width / 2, bb.y1 + 0.005, name,
                 fontsize=8, fontweight='bold', color=MSTYLE[name]['color'],
                 ha='center', va='bottom')

    # Section labels
    fig.canvas.draw()
    hm_bb = hm_axes[0].get_position()
    zm_bb = zm_axes[0].get_position()
    ms_bb = first_ms_ax.get_position()
    label_top_y = max(hm_bb.y1, zm_bb.y1) + 0.04
    # Align (a) and (c) titles at the same x
    title_x = ms_bb.x0
    fig.text(title_x, label_top_y,
             '(a) Distance matrices \u2014 Vortex Street',
             fontsize=10, fontweight='bold')
    fig.text(zm_bb.x0, label_top_y,
             '(b) Zoom: $t$=107\u2013110',
             fontsize=10, fontweight='bold', color='black')

    # ── Per-method colorbars to the right of each heatmap ──
    import math
    for idx, name in enumerate(methods):
        ax = hm_axes[idx]
        bb = ax.get_position()
        vmax = vmax_c[name]
        vmid = vmax / 2
        cb_w = 0.005
        cb_h = bb.height * 0.55
        cb_x = bb.x1 + 0.002
        cb_y = bb.y0 + (bb.height - cb_h) / 2
        cb_ax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
        cb = ColorbarBase(cb_ax, cmap=plt.get_cmap('viridis'),
                          norm=Normalize(0, vmax), orientation='vertical')
        cb.set_ticks([0, vmid, vmax])
        exp = math.floor(math.log10(vmax))
        scale = 10 ** exp
        labels = ['0', f'{vmid/scale:.1f}', f'{vmax/scale:.1f}']
        cb.set_ticklabels(labels)
        cb_bb = cb_ax.get_position()
        fig.text(cb_bb.x0 + cb_bb.width / 2 + 0.010, cb_bb.y1 + 0.003,
                 f'$\\times 10^{{{exp}}}$', fontsize=6, ha='center', va='bottom')
        cb_ax.tick_params(labelsize=6, length=1.0, width=0.2, pad=0.5)
        cb_ax.yaxis.set_ticks_position('right')
        cb.outline.set_linewidth(0.2)

    # "Distance" label to the right of last heatmap colorbar
    cb_bb = cb_ax.get_position()
    fig.text(cb_bb.x1 + 0.025, cb_bb.y0 + cb_bb.height / 2,
             'Distance', fontsize=7, ha='center',
             va='center', rotation=90, color='black')

    # ── (c) section label + CP legend + scalar colorbar ──
    label_y = ms_bb.y1 + 0.015
    fig.text(ms_bb.x0, label_y,
             '(c) MS complex',
             fontsize=10, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=CP_COLORS_FIG[k],
               markeredgecolor=CP_EDGES_FIG[k],
               markeredgewidth=0.8,
               markersize=5.5, label=CP_LABELS_FIG[k])
        for k in [0, 1, 2]
    ]
    leg_ax = fig.add_axes([0.18, label_y - 0.005, 0.25, 0.015])
    leg_ax.set_axis_off()
    leg_ax.legend(handles=legend_elements, loc='center left',
                  fontsize=8, ncol=3, framealpha=0.0,
                  borderpad=0, handletextpad=0.2,
                  columnspacing=0.5)

    copper_x0 = 0.48
    copper_w = 0.10
    cbar_ax = fig.add_axes([copper_x0, label_y - 0.001, copper_w, 0.006])
    cb = ColorbarBase(cbar_ax, cmap=_SCALAR_CMAP,
                      norm=Normalize(0, 1), orientation='horizontal')
    cb.set_ticks([])
    cb.outline.set_linewidth(0.2)
    fig.text(copper_x0 - 0.005, label_y + 0.003, 'Low',
             fontsize=7, ha='right', va='center')
    fig.text(copper_x0 + copper_w + 0.005, label_y + 0.003, 'High',
             fontsize=7, ha='left', va='center')
    fig.text(copper_x0 + copper_w / 2, label_y + 0.012, 'Speed',
             fontsize=7, ha='center', va='center', color='black')

    # ── Save ──
    out = root / "results/publication_figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        p = out / f"fig_vortexstreet_agreement.{ext}"
        fig.savefig(p)
        print(f"  Saved {p}")
    plt.close(fig)


if __name__ == '__main__':
    main()
