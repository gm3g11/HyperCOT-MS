#!/usr/bin/env python3
"""Combined Ionization Front figure — heatmaps + scalar field + region correspondence.

Single figure* (double-column, IEEE VIS / TVCG, 7.16" wide):
  Row 1:  (a) 4 distance-matrix heatmaps + zoom on t=3→4
  Row 2:  (b) scalar field t=3 (hillshade)  |  (c) scalar field t=4
  Row 3:  (d) MS regions t=3 (graph-colored) |  (e) t=4 regions (ξ-matched)

Color correspondence: panel (e) uses the SAME vivid palette as (d) via ξ argmax.
21/22 regions have >50% dominant match.  13/22 are off-diagonal (shifted with wave).
"""

import sys
import ast
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LightSource, PowerNorm, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.image import imread
from matplotlib.colors import to_rgba
from scipy.ndimage import laplace, gaussian_filter
import csv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mscoot.data_adapter import DatasetAdapter

# ══════════════════════════════════════════════════════════════════════════════
# Style
# ══════════════════════════════════════════════════════════════════════════════

DOUBLE_COL = 7.16
DS = 'ionization_front'
EVENT_T = 3

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
    'WD':      {'color': '#377eb8'},
    'GWD':     {'color': '#4daf4a'},
    'FGW':     {'color': '#ff7f00'},
    'MS-COOT': {'color': '#e41a1c'},
}
EVENT_COLOR = '#ff6600'

def make_unique_palette(n):
    """Generate n maximally-distinct colors using tab20 + extras."""
    cmap = plt.colormaps.get_cmap('tab20').resampled(20)
    colors = [cmap(i)[:3] for i in range(min(n, 20))]
    # If n > 20, append from Set3
    if n > 20:
        extra = plt.colormaps.get_cmap('Set3').resampled(12)
        for i in range(n - 20):
            colors.append(extra(i)[:3])
    return np.array(colors[:n])

CP_COLORS = {0: '#00BFFF', 1: 'white', 2: '#e41a1c'}

# Shared Oranges colormap — saturated floor like vortex street (0.25–0.85)
_base_oranges = plt.colormaps.get_cmap('Oranges')
SCALAR_CMAP = LinearSegmentedColormap.from_list(
    'trunc_oranges', _base_oranges(np.linspace(0.25, 0.85, 256)))
CP_LABELS = {0: 'Min', 1: 'Saddle', 2: 'Max'}


# ══════════════════════════════════════════════════════════════════════════════
# Adjacency & graph coloring
# ══════════════════════════════════════════════════════════════════════════════

def build_adjacency(hg_df):
    adj = defaultdict(set)
    cp_to_regions = defaultdict(set)
    for i, (_, row) in enumerate(hg_df.iterrows()):
        he = row['hyperedge']
        cps = ast.literal_eval(he) if isinstance(he, str) else he
        for cp in cps:
            cp_to_regions[cp].add(i)
    for _, regions in cp_to_regions.items():
        for r1 in regions:
            for r2 in regions:
                if r1 != r2:
                    adj[r1].add(r2)
    return adj


def graph_color(adj, n, n_palette=12):
    assignment = {}
    order = sorted(range(n), key=lambda r: len(adj.get(r, set())), reverse=True)
    for r in order:
        used = {assignment[nb] for nb in adj.get(r, set()) if nb in assignment}
        for c in range(n_palette):
            if c not in used:
                assignment[r] = c
                break
        else:
            assignment[r] = 0
    return assignment


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_matrix(path):
    with open(path) as f:
        c = f.read(1)
    return (pd.read_csv(path, index_col=0).values if c == ','
            else np.loadtxt(path, delimiter=','))


def load_all_matrices():
    dm = ROOT / "results/publication_figures/crc/results/ionization_front/distance_matrices"
    return {name: load_matrix(dm / fn) for name, fn in
            [('WD', 'wd_matrix.csv'), ('GWD', 'gwd_matrix.csv'),
             ('FGW', 'fgw_matrix.csv'), ('MS-COOT', 'coot_matrix.csv')]}


# ══════════════════════════════════════════════════════════════════════════════
# Segmentation + scalar field loading
# ══════════════════════════════════════════════════════════════════════════════

def load_segmentation_data(adapter, ts):
    config = adapter.config
    seg_raw = adapter.load_segmentation(ts)
    if len(seg_raw) == 0:
        return None

    cols = config.columns
    x_col = cols.get('x', 'Points:0')
    y_col = cols.get('y', 'Points:1')
    scalar_col = cols.get('scalar', 'Scalars_')

    for col_name, alts in [(x_col, ['Points:0', 'x']),
                            (y_col, ['Points:1', 'y']),
                            (scalar_col, ['Scalars_', 'scalar', 'data'])]:
        if col_name not in seg_raw.columns:
            for alt in alts:
                if alt in seg_raw.columns:
                    if alts == ['Points:0', 'x']:
                        x_col = alt
                    elif alts == ['Points:1', 'y']:
                        y_col = alt
                    else:
                        scalar_col = alt
                    break

    if x_col not in seg_raw.columns or y_col not in seg_raw.columns:
        return None

    x = seg_raw[x_col].values
    y = seg_raw[y_col].values
    scalar = (seg_raw[scalar_col].values if scalar_col in seg_raw.columns
              else np.zeros_like(x))
    msm = (seg_raw['MorseSmaleManifold'].values
           if 'MorseSmaleManifold' in seg_raw.columns else None)

    result = {'x': x, 'y': y, 'scalar': scalar, 'msm': msm,
              'coords_compatible': True}

    x_unique = np.sort(np.unique(np.round(x, 6)))
    y_unique = np.sort(np.unique(np.round(y, 6)))
    is_grid = False
    if len(x_unique) > 2 and len(y_unique) > 2:
        dx = np.median(np.diff(x_unique))
        dy = np.median(np.diff(y_unique))
        if (dx > 0 and dy > 0 and
                np.allclose(np.diff(x_unique), dx, rtol=0.02) and
                np.allclose(np.diff(y_unique), dy, rtol=0.02)):
            is_grid = True

    if is_grid:
        nx, ny = len(x_unique), len(y_unique)
        x_idx = np.clip(np.round((x - x_unique[0]) / dx).astype(int), 0, nx - 1)
        y_idx = np.clip(np.round((y - y_unique[0]) / dy).astype(int), 0, ny - 1)
        grid_msm = np.full((ny, nx), -1, dtype=int)
        grid_scalar = np.full((ny, nx), np.nan)
        grid_msm[y_idx, x_idx] = msm.astype(int) if msm is not None else -1
        grid_scalar[y_idx, x_idx] = scalar
        hdx, hdy = dx / 2, dy / 2
        result.update({
            'type': 'grid',
            'grid_msm': grid_msm,
            'grid_scalar': grid_scalar,
            'extent': [x_unique[0] - hdx, x_unique[-1] + hdx,
                       y_unique[0] - hdy, y_unique[-1] + hdy],
        })
    else:
        result['type'] = 'scatter'
    return result


def render_regions(ax, seg_data, rid_to_idx, color_array, blend_scalar=0.18,
                   edge_val=0.15):
    msm = seg_data['msm']
    unique_rids = np.unique(msm[msm >= 0])

    if seg_data['type'] == 'grid':
        grid_msm = seg_data['grid_msm']
        grid_scalar = seg_data['grid_scalar']
        h, w = grid_msm.shape
        rgb = np.full((h, w, 3), 0.90)
        for rid in unique_rids:
            ridx = rid_to_idx.get(int(rid))
            if ridx is None or ridx >= len(color_array):
                continue
            rgb[grid_msm == rid] = color_array[ridx]

        if blend_scalar > 0:
            valid = ~np.isnan(grid_scalar)
            smin, smax = np.nanmin(grid_scalar), np.nanmax(grid_scalar)
            gray_s = np.zeros_like(grid_scalar)
            gray_s[valid] = (grid_scalar[valid] - smin) / (smax - smin + 1e-12)
            for c in range(3):
                rgb[:, :, c] = np.where(
                    valid,
                    (1 - blend_scalar) * rgb[:, :, c] + blend_scalar * gray_s,
                    rgb[:, :, c])

        edges = laplace(grid_msm.astype(float))
        rgb[np.abs(edges) > 0.5] = [edge_val] * 3

        ax.imshow(rgb, aspect='auto', origin='lower', interpolation='nearest',
                  extent=seg_data['extent'], rasterized=True)


def add_cp_markers(ax, adapter, ts, markersize=2.5):
    cp = adapter.load_critical_points(ts)
    handles = []
    for ctype in [0, 1, 2]:
        sub = cp[cp['cp_type'] == ctype]
        if len(sub) > 0:
            h = ax.scatter(sub['x'].values, sub['y'].values,
                           marker='o', c=CP_COLORS[ctype],
                           s=markersize ** 2, edgecolors='white', linewidths=0.3,
                           zorder=5, label=CP_LABELS[ctype])
            handles.append(h)
    return handles


def compute_centroids(seg_data, rid_to_idx):
    centroids = {}
    if seg_data['type'] != 'grid':
        return centroids
    grid_msm = seg_data['grid_msm']
    extent = seg_data['extent']
    h, w = grid_msm.shape
    xx, yy = np.meshgrid(np.linspace(extent[0], extent[1], w),
                          np.linspace(extent[2], extent[3], h))
    for rid, idx in rid_to_idx.items():
        mask = grid_msm == rid
        if mask.any():
            centroids[idx] = (float(xx[mask].mean()), float(yy[mask].mean()))
    return centroids


# ══════════════════════════════════════════════════════════════════════════════
# Row 1: heatmaps
# ══════════════════════════════════════════════════════════════════════════════

def draw_heatmap(ax, mat, name, show_yticks=True, tick_step=None):
    nt = mat.shape[0]
    pos = mat[~np.eye(nt, dtype=bool)]
    pos = pos[pos > 0]
    vmax = np.percentile(pos, 97) if len(pos) else 1.0
    ax.imshow(mat, cmap='viridis', aspect='equal', vmin=0, vmax=vmax,
              interpolation='nearest', origin='upper', rasterized=True)

    # Method name as title above heatmap
    ax.set_title(name, fontsize=9, fontweight='bold',
                 color=MSTYLE[name]['color'], pad=2)

    if tick_step is None:
        tick_step = 60 if nt > 60 else 10
    ticks = list(range(0, nt, tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.tick_params(labelsize=7)
    if show_yticks:
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
    else:
        ax.set_yticks([])
    ax.set_box_aspect(1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
    return vmax


# ══════════════════════════════════════════════════════════════════════════════
# Row 2: scalar field (hillshade WarpByScalar effect)
# ══════════════════════════════════════════════════════════════════════════════

def draw_scalar_field(ax, seg_data, vmin, vmax):
    """Hillshaded scalar field — pseudo-3D WarpByScalar appearance."""
    grid = seg_data['grid_scalar'].copy()
    grid[np.isnan(grid)] = vmin
    ls = LightSource(azdeg=315, altdeg=35)
    rgb = ls.shade(grid, cmap=plt.cm.inferno, blend_mode='soft',
                   vert_exag=0.3, vmin=vmin, vmax=vmax)
    ax.imshow(rgb, aspect='auto', origin='lower', interpolation='bilinear',
              extent=seg_data['extent'], rasterized=True)


# ══════════════════════════════════════════════════════════════════════════════
# Row 2: 2D scalar field + MS complex overlay
# ══════════════════════════════════════════════════════════════════════════════

def _build_polylines(sep_df):
    """Build polylines from TTK separatrix geometry CSV."""
    lines = []
    for sid in sep_df['SeparatrixId'].unique():
        seg = sep_df[sep_df.SeparatrixId == sid].sort_values('PointOrder')
        pts = seg[['Points:0', 'Points:1']].values
        if len(pts) >= 2:
            lines.append(pts)
    return lines


def load_cps_and_seps(ts):
    """Load CPs + TTK separatrix polylines. Reconstruct missing CPs from endpoints."""
    ttk_dir = ROOT / f'datasets/{DS}/ttk_output'

    # Load CPs
    cps = []
    cp_cellids = set()
    with open(ttk_dir / 'critical_points.csv') as f:
        for row in csv.DictReader(f):
            if int(row['TimeStep']) == ts:
                cps.append({'type': int(row['CellDimension']),
                            'x': float(row['Points:0']),
                            'y': float(row['Points:1']),
                            'cell_id': int(row['CellId'])})
                cp_cellids.add(int(row['CellId']))

    # Load TTK separatrix geometry
    sep_path = ttk_dir / f'separatrix_geometry_{ts}.csv'
    sep_df = pd.read_csv(sep_path)
    polylines = _build_polylines(sep_df)

    # Reconstruct missing CPs from separatrix endpoints
    seen = set()
    for sid in sep_df['SeparatrixId'].unique():
        seg = sep_df[sep_df.SeparatrixId == sid].sort_values('PointOrder')
        pts = seg[['Points:0', 'Points:1']].values
        src = int(seg.iloc[0]['SourceId'])
        dst = int(seg.iloc[0]['DestinationId'])
        stype = int(seg.iloc[0]['SeparatrixType'])
        # Source is always a saddle
        if src not in cp_cellids and src not in seen:
            cps.append({'type': 1, 'x': pts[0, 0], 'y': pts[0, 1], 'cell_id': src})
            seen.add(src)
        # Destination: min for type 0, max for type 1
        if dst not in cp_cellids and dst not in seen:
            dim = 0 if stype == 0 else 2
            cps.append({'type': dim, 'x': pts[-1, 0], 'y': pts[-1, 1], 'cell_id': dst})
            seen.add(dst)

    if seen:
        print(f'  Reconstructed {len(seen)} missing CPs for t={ts}')

    return cps, polylines


MS_CP_STYLE = {
    0: {'color': '#377eb8', 'edge': '#1a4a7a', 'size': 70, 'label': 'min'},
    1: {'color': 'white', 'edge': '#444444', 'size': 45, 'label': 'saddle'},
    2: {'color': '#e41a1c', 'edge': '#a01015', 'size': 60, 'label': 'max'},
}


def draw_ms_overlay(ax, seg_data, cps, sep_polylines, highlight_rids=None,
                    highlight_colors=None, crop_x=300,
                    show_boundary=True, transpose=False,
                    crop_xlim=None, crop_ylim=None):
    """Draw scalar field + TTK separatrices + CPs + highlights."""
    grid = seg_data['grid_scalar'].copy()
    msm = seg_data['grid_msm']
    extent = list(seg_data['extent'])

    if transpose:
        grid = grid.T
        msm = msm.T
        extent = [extent[2], extent[3], extent[0], extent[1]]
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    grid[np.isnan(grid)] = vmin

    # Scalar field
    norm = PowerNorm(gamma=0.45, vmin=vmin, vmax=vmax)
    ax.imshow(grid, cmap=SCALAR_CMAP, aspect='equal', origin='lower',
              interpolation='bilinear', extent=extent, norm=norm,
              rasterized=True, zorder=0, alpha=1.0)

    h, w = grid.shape
    xs = np.linspace(extent[0], extent[1], w)
    ys = np.linspace(extent[2], extent[3], h)

    # TTK separatrices (clipped to view if crop limits given)
    if show_boundary and sep_polylines is not None:
        for pl in sep_polylines:
            px = pl[:, 1] if transpose else pl[:, 0]
            py = pl[:, 0] if transpose else pl[:, 1]
            ax.plot(px, py, 'k-', lw=0.8, zorder=2,
                    solid_capstyle='round', clip_on=True)

    # Highlighted regions: bold outline only (no fill)
    if highlight_rids and highlight_colors:
        for rid, hc in zip(highlight_rids, highlight_colors):
            mask_s = gaussian_filter((msm == rid).astype(float), sigma=0.5)
            ax.contour(xs, ys, mask_s, levels=[0.5], colors=[hc],
                       linewidths=1.8, linestyles='solid', zorder=5)

    # CPs — min/max first, saddles on top
    for cp_type in [0, 2, 1]:
        typed = [c for c in cps if c['type'] == cp_type]
        if not typed:
            continue
        if transpose:
            cx = [c['y'] for c in typed]
            cy = [c['x'] for c in typed]
        else:
            cx = [c['x'] for c in typed]
            cy = [c['y'] for c in typed]
        s = MS_CP_STYLE[cp_type]
        z = 7 if cp_type == 1 else 6  # saddles on top
        ax.scatter(cx, cy, c=s['edge'], marker='o', s=s['size'] * 1.3,
                   edgecolors='none', alpha=0.3, zorder=z)
        ax.scatter(cx, cy, c=s['color'], marker='o', s=s['size'],
                   edgecolors=s['edge'], linewidths=0.8, zorder=z + 1)

    if not transpose:
        ax.set_xlim(extent[0], crop_x)
    ax.set_xticks([]); ax.set_yticks([])


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    adapter = DatasetAdapter(ROOT / f'datasets/{DS}')

    # ── Load data ──────────────────────────────────────────────────────────
    print('Loading distance matrices...')
    matrices = load_all_matrices()

    print('Loading hypergraphs...')
    hyper_dir = ROOT / f'results/{DS}/hypergraphs'
    hg1 = pd.read_csv(hyper_dir / f'hypergraph_{EVENT_T:03d}.csv')
    hg2 = pd.read_csv(hyper_dir / f'hypergraph_{EVENT_T + 1:03d}.csv')
    rid_to_idx1 = {int(r['region_id']): i
                   for i, (_, r) in enumerate(hg1.iterrows())}
    rid_to_idx2 = {int(r['region_id']): i
                   for i, (_, r) in enumerate(hg2.iterrows())}
    n_src, n_tgt = len(hg1), len(hg2)
    print(f'  t={EVENT_T}: {n_src} regions  |  t={EVENT_T + 1}: {n_tgt} regions')

    print('Loading xi coupling...')
    xi_path = ROOT / f'results/{DS}/couplings/xi_{EVENT_T:03d}_{EVENT_T + 1:03d}.csv'
    xi = pd.read_csv(xi_path, header=0, index_col=None).values.astype(float)
    if xi.shape[0] > n_src:
        xi = xi[1:]
    xi = xi[:n_src, :n_tgt]

    # Hungarian 1:1 matching on raw xi
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-xi)

    # Compute centroids for spatial coherence check
    seg_all = pd.read_csv(ROOT / f'datasets/{DS}/ttk_output/segmentation.csv')
    def get_centroids(ts, hg):
        sub = seg_all[seg_all['TimeStep'] == ts]
        x, y = sub['Points:0'].values, sub['Points:1'].values
        msm = sub['MorseSmaleManifold'].values
        centroids = {}
        for idx in range(len(hg)):
            rid = int(hg.iloc[idx]['region_id'])
            mask = msm == rid
            if mask.sum() > 0:
                centroids[idx] = (x[mask].mean(), y[mask].mean())
        return centroids

    src_centroids = get_centroids(EVENT_T, hg1)
    tgt_centroids = get_centroids(EVENT_T + 1, hg2)

    src_colors = make_unique_palette(n_src)
    NEUTRAL = np.array([0.82, 0.78, 0.72])  # warm beige for unmatched
    tgt_colors = np.full((n_tgt, 3), 0.85)
    tgt_claimed_by = {}
    DIST_THRESH = 40  # only color-match if centroids < 40 apart

    n_matched = 0
    for i, j in zip(row_ind, col_ind):
        if xi[i, j] <= 0:
            continue
        if i in src_centroids and j in tgt_centroids:
            sc = src_centroids[i]; tc = tgt_centroids[j]
            dist = np.sqrt((sc[0] - tc[0]) ** 2 + (sc[1] - tc[1]) ** 2)
        else:
            dist = 999
        if dist < DIST_THRESH:
            tgt_colors[j] = src_colors[i]
            tgt_claimed_by[j] = (i, dist)
            n_matched += 1
        else:
            # Spatially incoherent — use neutral
            tgt_colors[j] = NEUTRAL
            src_colors[i] = NEUTRAL  # also neutralize in panel (d)

    n_diag = sum(1 for j, (si, _) in tgt_claimed_by.items() if si == j)
    n_shifted = n_matched - n_diag
    print(f'  Spatially coherent: {n_matched}/{n_tgt}, diagonal: {n_diag}, shifted: {n_shifted}')

    # ── Load segmentation ──────────────────────────────────────────────────
    print('Loading segmentation...')
    seg1 = load_segmentation_data(adapter, EVENT_T)
    seg2 = load_segmentation_data(adapter, EVENT_T + 1)
    if seg1 is None or seg2 is None:
        print('ERROR: segmentation data not found.')
        return

    centroids1 = compute_centroids(seg1, rid_to_idx1)
    centroids2 = compute_centroids(seg2, rid_to_idx2)

    # Shared scalar range for both panels
    s1 = seg1['grid_scalar']; s2 = seg2['grid_scalar']
    scalar_min = min(np.nanmin(s1), np.nanmin(s2))
    scalar_max = max(np.nanmax(s1), np.nanmax(s2))

    # ══════════════════════════════════════════════════════════════════════
    # Figure — 3 column groups: (a+b) left, (c,d) middle, (e,f) right
    # ══════════════════════════════════════════════════════════════════════
    print('Building figure...')
    import matplotlib.patheffects as pe
    from matplotlib.patches import Ellipse

    fig = plt.figure(figsize=(DOUBLE_COL, 3.4), facecolor='white')

    N_TS = 31  # only show timesteps 0–30
    BOX_LO, BOX_HI = 0, 8
    ZOOM_LO, ZOOM_HI = 2, 6
    zr = slice(ZOOM_LO, ZOOM_HI)

    # ── Left column: (a) Heatmaps 2×2 + (b) Zoom 1×2 below ──────────────
    gs_left = GridSpec(3, 2, figure=fig,
                       left=0.04, right=0.34,
                       top=0.93, bottom=0.03,
                       wspace=0.40, hspace=0.16,
                       height_ratios=[1, 1, 0.50])

    methods = ['WD', 'GWD', 'FGW', 'MS-COOT']
    hm_pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    hm_axes = []
    hm_vmax = {}
    for (r, c), name in zip(hm_pos, methods):
        ax = fig.add_subplot(gs_left[r, c])
        hm_axes.append(ax)
        mat_sub = matrices[name][:N_TS, :N_TS]
        hm_vmax[name] = draw_heatmap(ax, mat_sub, name, show_yticks=(c == 0), tick_step=10)
        # Arrow pointing to zoom region on FGW and MS-COOT heatmaps
        if name in ('FGW', 'MS-COOT'):
            mid = (BOX_LO + BOX_HI) / 2
            ax.annotate('', xy=(mid, mid), xytext=(mid + 7, mid),
                        arrowprops=dict(arrowstyle='->', color='#ff3333', lw=1.2),
                        zorder=4)
    # Axis labels — "Timestep (t)" on y-axes, no x-axis label
    hm_axes[0].set_ylabel('Timestep (t)', fontsize=8, labelpad=1)
    hm_axes[2].set_ylabel('Timestep (t)', fontsize=8, labelpad=1)

    # (b) Zoom panels — row 3 of left column
    zoom_axes = []
    for i, nm in enumerate(['FGW', 'MS-COOT']):
        ax = fig.add_subplot(gs_left[2, i])
        zoom_axes.append(ax)
        mt = matrices[nm]
        sub = mt[zr, zr]
        ns = sub.shape[0]
        off_sub = sub[~np.eye(ns, dtype=bool)]
        vx_local = off_sub.max() if off_sub.max() > 0 else 1e-6
        ax.imshow(sub, cmap='viridis', aspect='equal', vmin=0, vmax=vx_local,
                  interpolation='nearest', origin='upper', rasterized=True,
                  extent=[ZOOM_LO - 0.5, ZOOM_HI - 0.5,
                          ZOOM_HI - 0.5, ZOOM_LO - 0.5])
        ax.add_patch(Rectangle((EVENT_T + 0.5, EVENT_T - 0.5), 1, 1,
                                lw=1.5, ec='#ff3333', fc='none', zorder=5))
        ax.set_title(f'Zoom: {nm}', fontsize=8, fontweight='bold',
                     color=MSTYLE[nm]['color'], pad=2)
        ax.set_xticks(range(ZOOM_LO, ZOOM_HI))
        ax.set_yticks(range(ZOOM_LO, ZOOM_HI) if i == 0 else [])
        ax.tick_params(labelsize=7)
        for sp in ax.spines.values():
            sp.set_linewidth(1.5); sp.set_edgecolor('#ff3333')
        ax.set_box_aspect(1)

    # Individual colorbars for each heatmap — with real distance values
    fig.canvas.draw()
    for idx, (name, ax_hm) in enumerate(zip(methods, hm_axes)):
        pos = ax_hm.get_position()
        vmax = hm_vmax[name]
        cb_ax = fig.add_axes([pos.x1 + 0.008, pos.y0 + pos.height * 0.15,
                               0.004, pos.height * 0.55])
        cb = ColorbarBase(cb_ax, cmap=plt.get_cmap('viridis'),
                           norm=Normalize(0, vmax), orientation='vertical')
        # Format: two-digit coefficient, ×10^n aligned above colorbar
        if vmax > 0:
            exp = int(np.floor(np.log10(vmax)))
            coeff = vmax / 10**exp
            vmax_str = f'{coeff:.1f}'
            cb.set_ticks([0, vmax])
            cb.set_ticklabels(['0', vmax_str])
            # ×10^n label — right and up via fig.text
            cb_pos2 = cb.ax.get_position()
            fig.text(cb_pos2.x1 + 0.001, cb_pos2.y1 + 0.012,
                     r'$\times 10^{' + str(exp) + r'}$',
                     fontsize=7, va='bottom', ha='left')
        else:
            cb.set_ticks([0])
            cb.set_ticklabels(['0'])
        cb.ax.tick_params(labelsize=7, width=0.2, length=1.0, pad=1)
        cb.outline.set_linewidth(0.2)

        # "Distance" rotated label on GWD and MS-COOT colorbars only
        if name in ('GWD', 'MS-COOT'):
            cb_pos3 = cb.ax.get_position()
            fig.text(cb_pos3.x1 + 0.012, cb_pos3.y0 + cb_pos3.height / 2,
                     'Distance', fontsize=7, rotation=90,
                     ha='left', va='center', color='black', fontstyle='italic')

    # Left column titles
    fig.text(0.04, 0.96, '(a)', fontsize=10, fontweight='bold')
    fig.text(0.07, 0.96, 'Distance matrices \u2014 Ionization Front', fontsize=9)


    # ── Middle column: (c,d) Scalar field ─────────────────────────────────
    _warp_ec = '#00b380'
    warp_dir = ROOT / 'results' / DS / 'figures'
    warp3 = imread(str(warp_dir / f'warp_ts{EVENT_T:03d}.png'))
    warp4 = imread(str(warp_dir / f'warp_ts{EVENT_T + 1:03d}.png'))

    def find_content_bbox(img, thresh=0.95):
        mask = np.any(img[:, :, :3] < thresh, axis=2)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
        cmin, cmax = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
        return rmin, rmax, cmin, cmax

    bb3 = find_content_bbox(warp3)
    bb4 = find_content_bbox(warp4)
    rmin = min(bb3[0], bb4[0]); rmax = max(bb3[1], bb4[1])
    cmin = min(bb3[2], bb4[2]); cmax = max(bb3[3], bb4[3])
    h_img, w_img = warp3.shape[:2]
    rmin = max(0, rmin + 40)
    rmax = min(h_img, rmax - 30)
    cmin = max(0, cmin + 350)
    cmax = min(w_img, cmax - 380)
    warp3 = warp3[rmin:rmax, cmin:cmax]
    warp4 = warp4[rmin:rmax, cmin:cmax]

    gs_mid = GridSpec(2, 1, figure=fig,
                      left=0.42, right=0.68,
                      top=0.87, bottom=0.04,
                      hspace=0.14)

    ax_c = fig.add_subplot(gs_mid[0, 0])
    ax_c.set_facecolor('white')
    ax_c.imshow(warp3, aspect='auto', interpolation='bilinear')
    ax_c.set_xticks([]); ax_c.set_yticks([])
    for sp in ax_c.spines.values():
        sp.set_linewidth(0.4); sp.set_edgecolor('0.5')

    ax_d = fig.add_subplot(gs_mid[1, 0])
    ax_d.set_facecolor('white')
    ax_d.imshow(warp4, aspect='auto', interpolation='bilinear')
    ax_d.set_xticks([]); ax_d.set_yticks([])
    for sp in ax_d.spines.values():
        sp.set_linewidth(0.4); sp.set_edgecolor('0.5')

    for ax_warp in [ax_c, ax_d]:
        ax_warp.add_patch(Ellipse(
            (0.42, 0.50), 0.35, 0.40,
            transform=ax_warp.transAxes,
            lw=1.8, ec=_warp_ec, fc='none', ls='--', zorder=10,
            clip_on=True))
    ax_d.text(0.42, 0.78, 'merged', transform=ax_d.transAxes,
              fontsize=10, color=_warp_ec, ha='center', fontweight='bold',
              fontstyle='italic',
              path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])

    ax_c.text(0.03, 0.92, f't = {EVENT_T}', transform=ax_c.transAxes,
              fontsize=8, fontweight='bold', va='top', color='white',
              bbox=dict(boxstyle='round,pad=0.08', fc='0.3', ec='none', alpha=0.8))
    ax_d.text(0.03, 0.92, f't = {EVENT_T + 1}', transform=ax_d.transAxes,
              fontsize=8, fontweight='bold', va='top', color='white',
              bbox=dict(boxstyle='round,pad=0.08', fc='0.3', ec='none', alpha=0.8))

    # Arrow t=3 → t=4 in (b) scalar field
    ax_c.text(0.5, -0.08, r'$\downarrow$', transform=ax_c.transAxes,
              fontsize=16, ha='center', va='center', color='0.2',
              fontweight='bold', clip_on=False)

    fig.text(0.42, 0.96, '(b)', fontsize=10, fontweight='bold')
    fig.text(0.45, 0.96, 'Scalar field', fontsize=9)

    # ── Right column: (e,f) MS complex ────────────────────────────────────
    print('Loading CPs and separatrices...')
    cps3, seps3 = load_cps_and_seps(EVENT_T)
    cps4, seps4 = load_cps_and_seps(EVENT_T + 1)

    MS_XLIM = (135, 275)
    MS_YLIM = (78, 178)

    gs_right = GridSpec(2, 1, figure=fig,
                        left=0.70, right=0.98,
                        top=0.87, bottom=0.04,
                        hspace=0.14)

    ax_e = fig.add_subplot(gs_right[0, 0])
    ax_e.set_facecolor('#fce8d0')
    draw_ms_overlay(ax_e, seg1, cps3, seps3, crop_x=MS_XLIM[1],
                    show_boundary=True,
                    crop_xlim=MS_XLIM, crop_ylim=MS_YLIM)
    ax_e.set_xlim(*MS_XLIM); ax_e.set_ylim(*MS_YLIM)
    ax_e.set_aspect('equal', adjustable='datalim')
    for sp in ax_e.spines.values():
        sp.set_visible(False)

    ax_f = fig.add_subplot(gs_right[1, 0])
    ax_f.set_facecolor('#fce8d0')
    draw_ms_overlay(ax_f, seg2, cps4, seps4, crop_x=MS_XLIM[1],
                    show_boundary=True,
                    crop_xlim=MS_XLIM, crop_ylim=MS_YLIM)
    ax_f.set_xlim(*MS_XLIM); ax_f.set_ylim(*MS_YLIM)
    ax_f.set_aspect('equal', adjustable='datalim')
    for sp in ax_f.spines.values():
        sp.set_visible(False)

    for ax_ms in [ax_e, ax_f]:
        ax_ms.add_patch(Ellipse(
            (218, 128), 65, 45,
            lw=1.5, ec=_warp_ec, fc='none', ls='--', zorder=8))

    ax_e.text(0.03, 0.92, f't = {EVENT_T}', transform=ax_e.transAxes,
              fontsize=8, fontweight='bold', va='top', color='0.2',
              bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='0.4',
                        lw=0.3, alpha=0.8))
    ax_f.text(0.03, 0.92, f't = {EVENT_T + 1}', transform=ax_f.transAxes,
              fontsize=8, fontweight='bold', va='top', color='0.2',
              bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='0.4',
                        lw=0.3, alpha=0.8))

    # Arrow t=3 → t=4 in (c) MS complex
    ax_e.text(0.5, -0.08, r'$\downarrow$', transform=ax_e.transAxes,
              fontsize=16, ha='center', va='center', color='0.2',
              fontweight='bold', clip_on=False)

    fig.text(0.70, 0.96, '(c)', fontsize=10, fontweight='bold')
    fig.text(0.73, 0.96, 'MS complex', fontsize=9)

    # ── Shared legends — centered at top between (b) and (c) ─────────
    # Scalar colorbar
    scbar_ax = fig.add_axes([0.52, 0.915, 0.10, 0.008])
    scbar = ColorbarBase(scbar_ax, cmap=SCALAR_CMAP,
                          norm=Normalize(0, 1), orientation='horizontal')
    scbar.set_ticks([])
    scbar.outline.set_linewidth(0.3)
    fig.text(0.515, 0.918, 'Low', fontsize=7, ha='right', va='center')
    fig.text(0.57, 0.928, 'Density', fontsize=7, ha='center', va='bottom',
             fontstyle='italic', color='black')
    fig.text(0.625, 0.918, 'High', fontsize=7, ha='left', va='center')

    # CP legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=MS_CP_STYLE[k]['color'],
               markeredgecolor=MS_CP_STYLE[k]['edge'],
               markersize=5, label=MS_CP_STYLE[k]['label'])
        for k in [0, 1, 2]
    ]
    fig.legend(handles=legend_elements, loc='center',
               fontsize=7, ncol=3, framealpha=0.0,
               bbox_to_anchor=(0.84, 0.918),
               borderpad=0, handletextpad=0.2,
               columnspacing=0.5)

    # ══════════════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════════════
    out = ROOT / "results/publication_figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        p = out / f'fig_ionfront_combined.{ext}'
        fig.savefig(p, facecolor='white')
        print(f'  Saved {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()
