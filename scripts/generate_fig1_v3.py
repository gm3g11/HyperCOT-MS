#!/usr/bin/env python3
"""Generate fig1_method_overview_v3.

Layout: 2x4 grid
  Col 0: Scalar fields (3D views)
  Col 1: MS complexes (TTK rendered)
  Col 2: Region correspondence (xi) — hypergraph regions colored by coupling
  Col 3: CP correspondence (pi) — MCOpt-style color transfer on Morse graph
         (both nodes AND edges colored by correspondence)
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ── IEEE VIS / TVCG Style ──────────────────────────────────────────────────
SINGLE_COL = 3.33
DOUBLE_COL = 7.0

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.framealpha': 0.85,
    'legend.edgecolor': '0.7',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

CP_MARKERS = {0: 'o', 1: 'o', 2: 'o'}
CP_COLORS_MAP = {0: '#3745a4', 1: '#FFFFFF', 2: '#e41a1c'}
CP_COLORS_LEGEND = CP_COLORS_MAP  # same colors for legend
CP_LABELS = {0: 'Min', 1: 'Saddle', 2: 'Max'}

# 12 saturated, maximally distinct colors for region fills.
REGION_PALETTE = [
    '#e6194B',  # red
    '#3cb44b',  # green
    '#4363d8',  # blue
    '#f58231',  # orange
    '#911eb4',  # purple
    '#42d4f4',  # cyan
    '#f032e6',  # magenta
    '#bfef45',  # lime
    '#469990',  # teal
    '#9A6324',  # brown
    '#800000',  # maroon
    '#000075',  # navy
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def build_region_adjacency(hyper_df):
    """Two regions are adjacent if they share any CP (min, max, or saddle)."""
    adj = defaultdict(set)
    cp_to_regions = defaultdict(list)
    for i, row in hyper_df.iterrows():
        cps = {int(row['min_id']), int(row['max_id'])}
        for s in eval(row['boundary_saddles']):
            cps.add(int(s))
        for cp in cps:
            cp_to_regions[cp].append(i)
    for regions in cp_to_regions.values():
        for a in regions:
            for b in regions:
                if a != b:
                    adj[a].add(b)
    return adj


def graph_coloring(n, adj, palette_hex):
    """Greedy graph coloring that maximises palette diversity."""
    palette = [_hex_to_rgb(h) for h in palette_hex]
    n_pal = len(palette)
    assignment = [None] * n
    order = sorted(range(n), key=lambda i: -len(adj.get(i, set())))

    for i in order:
        used = {assignment[j] for j in adj.get(i, set())
                if assignment[j] is not None}
        available = [c for c in range(n_pal) if c not in used]
        if not available:
            available = list(range(n_pal))
        counts = defaultdict(int)
        for c in assignment:
            if c is not None:
                counts[c] += 1
        assignment[i] = min(available, key=lambda c: counts[c])

    return [palette[c] for c in assignment]


def _crop_paraview_img(img, crop_right_frac=0.16, auto_trim=True,
                       replace_bg_white=False):
    """Crop ParaView legend bars and optionally replace gray background."""
    h, w = img.shape[:2]
    right = int(w * (1 - crop_right_frac))
    cropped = img[:, :right].copy()
    bl_size = max(35, int(h * 0.06))
    bg = cropped[5, 5]
    cropped[h - bl_size:, :] = bg

    if auto_trim and cropped.ndim == 3:
        bg_color = cropped[2, 2, :3]

        if replace_bg_white:
            diff_bg = np.abs(cropped[:, :, :3].astype(float)
                             - bg_color.astype(float))
            thr = 12.0 / 255.0 if cropped.dtype != np.uint8 else 12.0
            bg_mask = diff_bg.max(axis=2) <= thr
            if cropped.dtype == np.uint8:
                cropped[bg_mask, :3] = 255
                if cropped.shape[2] == 4:
                    cropped[bg_mask, 3] = 255
            else:
                cropped[bg_mask, :3] = 1.0
            bg_color = cropped[2, 2, :3]

        diff = np.abs(cropped[:, :, :3].astype(float)
                      - bg_color.astype(float))
        mask = diff.max(axis=2) > 15
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            pad = 4
            rmin = max(0, rmin - pad)
            rmax = min(cropped.shape[0], rmax + pad + 1)
            cmin = max(0, cmin - pad)
            cmax = min(cropped.shape[1], cmax + pad + 1)
            cropped = cropped[rmin:rmax, cmin:cmax]
    return cropped


def _extract_graph_edges(cp_df, sep_df):
    """Extract unique graph edges from separatrices as (src_pos, dst_pos) pairs.

    Maps CellId -> row position in cp_df, handling the TTK CellId duplicate bug
    by keying on (CellId, CellDimension).
    """
    cellid_dim_to_pos = {}
    for pos in range(len(cp_df)):
        row = cp_df.iloc[pos]
        key = (int(row['CellId']), int(row['CellDimension']))
        if key not in cellid_dim_to_pos:
            cellid_dim_to_pos[key] = pos

    edges_df = sep_df.groupby('SeparatrixId').first()[
        ['SourceId', 'DestinationId', 'SeparatrixType']]

    edges = []
    for _, e in edges_df.iterrows():
        src_key = (int(e['SourceId']), 1)  # source = saddle
        dst_dim = 0 if int(e['SeparatrixType']) == 0 else 2
        dst_key = (int(e['DestinationId']), dst_dim)
        if src_key in cellid_dim_to_pos and dst_key in cellid_dim_to_pos:
            edges.append((cellid_dim_to_pos[src_key],
                          cellid_dim_to_pos[dst_key]))
    return edges


def _draw_cp_correspondence_panel(ax, cp_df, edges, cp_colors,
                                   node_size=30, edge_lw=1.8, edge_alpha=0.75,
                                   node_edge_lw=0.4):
    """Draw Morse graph with MCOpt-style color transfer.

    Both CPs AND edges are colored by correspondence — the key visual from
    Li et al. (MCOpt, IEEE VIS 2023, Fig. 2).

    cp_colors: list of RGB tuples, one per row in cp_df.
    edges: list of (src_pos, dst_pos) tuples from _extract_graph_edges().
    """
    # Draw edges colored by endpoint average
    for si, di in edges:
        sx, sy = cp_df.iloc[si]['Points_0'], cp_df.iloc[si]['Points_1']
        dx, dy = cp_df.iloc[di]['Points_0'], cp_df.iloc[di]['Points_1']
        c1 = np.array(cp_colors[si][:3])
        c2 = np.array(cp_colors[di][:3])
        edge_col = tuple((c1 + c2) / 2)
        ax.plot([sx, dx], [sy, dy], color=edge_col, linewidth=edge_lw,
                alpha=edge_alpha, zorder=1, solid_capstyle='round')

    # Draw CPs — all circles, color = correspondence
    for pos in range(len(cp_df)):
        row = cp_df.iloc[pos]
        ax.scatter(row['Points_0'], row['Points_1'],
                   c=[cp_colors[pos][:3]], marker='o',
                   s=node_size, zorder=5,
                   edgecolors='black', linewidths=node_edge_lw)

    ax.set_xlim(-8, 108)
    ax.set_ylim(-8, 108)
    ax.set_aspect('equal')
    ax.axis('off')


def _draw_xi_correspondence_panel(ax, cp_df, hyper_df, colors, opacity_map=None,
                                   label_map=None, label_nudges=None,
                                   fill_regions=False, fill_alpha=0.25,
                                   show_labels=True, min_label_area=15,
                                   label_fontsize=6,
                                   edge_linewidth=0.8, edge_color_uniform=None):
    """Draw region polygons colored by correspondence on a single axis."""
    centroids = {}
    label_candidates = []

    for i, row in hyper_df.iterrows():
        min_id = int(row['min_id'])
        max_id = int(row['max_id'])
        saddle_ids = eval(row['boundary_saddles'])
        coords = []
        for cp_id in [min_id] + list(saddle_ids) + [max_id]:
            coords.append([cp_df.iloc[cp_id]['Points_0'],
                           cp_df.iloc[cp_id]['Points_1']])
        coords = np.array(coords)
        cx, cy = coords.mean(axis=0)
        centroids[i] = (cx, cy)
        angles = np.arctan2(coords[:, 1] - cy, coords[:, 0] - cx)
        vertices = coords[np.argsort(angles)]

        alpha = opacity_map[i] if opacity_map and i in opacity_map else 0.7
        color = colors[i][:3] if len(colors[i]) >= 3 else colors[i]
        if fill_regions and alpha > 0:
            facecolor = (*color, fill_alpha)
        else:
            facecolor = 'none'
        if edge_color_uniform is not None:
            edgecolor = edge_color_uniform
        else:
            edgecolor = (*color, max(alpha, 0.5)) if alpha > 0 else (0.6, 0.6, 0.6, 0.4)

        poly = MplPolygon(vertices, closed=True, facecolor=facecolor,
                          edgecolor=edgecolor, linewidth=edge_linewidth)
        ax.add_patch(poly)

        if not show_labels:
            continue

        n_v = len(vertices)
        area = 0.5 * abs(sum(vertices[j][0] * vertices[(j+1) % n_v][1]
                              - vertices[(j+1) % n_v][0] * vertices[j][1]
                              for j in range(n_v)))

        if alpha > 0 and area > min_label_area:
            label = label_map[i] if label_map and i in label_map else str(i + 1)
            if label:
                label_candidates.append((cx, cy, label, area, i))

    # ── Place labels with multi-direction collision avoidance ────────────
    if show_labels and label_candidates:
        label_candidates.sort(key=lambda t: -t[3])

        deduped = []
        for cand in label_candidates:
            cx_, cy_, lbl_, area_, _ = cand
            skip = False
            for kept in deduped:
                if kept[2] == lbl_:
                    d = np.sqrt((cx_ - kept[0])**2 + (cy_ - kept[1])**2)
                    if d < 15:
                        skip = True
                        break
            if not skip:
                deduped.append(cand)
        label_candidates = deduped

        all_cp_xy = np.array([(cp_df.iloc[k]['Points_0'],
                               cp_df.iloc[k]['Points_1'])
                              for k in range(len(cp_df))])
        min_sep = 6.0
        offsets = [(0, 0)]
        for r in (3.5, 6.0, 9.0, 12.0):
            offsets.extend([(r, 0), (-r, 0), (0, r), (0, -r),
                            (r*0.7, r*0.7), (-r*0.7, r*0.7),
                            (r*0.7, -r*0.7), (-r*0.7, -r*0.7),
                            (r, r*0.4), (-r, r*0.4),
                            (r, -r*0.4), (-r, -r*0.4)])

        placed = []
        for cx, cy, label, area, _ in label_candidates:
            best_score, best_lx, best_ly = -1e9, cx, cy
            for dx, dy in offsets:
                lx, ly = cx + dx, cy + dy
                if lx < -8 or lx > 108 or ly < -8 or ly > 108:
                    continue
                score = 0.0
                for px, py in placed:
                    d = np.sqrt((lx - px)**2 + (ly - py)**2)
                    if d < min_sep:
                        score -= 5.0 * (min_sep - d)
                cp_dists = np.sqrt((all_cp_xy[:, 0] - lx)**2 +
                                   (all_cp_xy[:, 1] - ly)**2)
                close_cps = cp_dists[cp_dists < 4.0]
                score -= np.sum(2.0 * (4.0 - close_cps))
                score -= 0.08 * np.sqrt(dx**2 + dy**2)
                if score > best_score:
                    best_score = score
                    best_lx, best_ly = lx, ly

            if label_nudges and label in label_nudges:
                best_lx += label_nudges[label][0]
                best_ly += label_nudges[label][1]
            placed.append((best_lx, best_ly))
            ax.text(best_lx, best_ly, label, ha='center', va='center',
                    fontsize=label_fontsize, fontweight='bold', color='black',
                    zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.0,
                                                foreground='white')])

    for cp_type in [0, 2, 1]:  # saddles last so white dots render on top
        mask = cp_df['CellDimension'] == cp_type
        lw = 0.8 if cp_type == 1 else 0.4  # thicker outline for white saddles
        ax.scatter(cp_df.loc[mask, 'Points_0'], cp_df.loc[mask, 'Points_1'],
                   c=CP_COLORS_MAP[cp_type], marker=CP_MARKERS[cp_type],
                   s=30, zorder=5, edgecolors='black', linewidths=lw,
                   alpha=0.90)

    ax.set_xlim(-8, 108)
    ax.set_ylim(-8, 108)
    ax.set_aspect('equal')
    ax.axis('off')
    return centroids


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    root = Path(__file__).resolve().parent.parent
    sin_dir = root / 'results' / 'sinusoidal'
    fig_src = sin_dir / 'figures'
    hyper_dir = sin_dir / 'hypergraphs'
    couplings_dir = sin_dir / 'couplings'
    ttk_dir = root / 'datasets' / 'sinusoidal' / 'ttk_output'

    fig_dir = root / 'results' / 'publication_figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load images ─────────────────────────────────────────────────────────
    from PIL import Image as PILImage

    img_files = {
        'clean_3d': fig_src / 'clean_input_ori.png',
        'noisy_3d': fig_src / 'noise_input_ori.png',
        'clean_ms': fig_src / 'clean_input.png',
        'noisy_ms': fig_src / 'noise_input.png',
    }
    for name, path in img_files.items():
        if not path.exists():
            print(f'Missing {path}')
            return

    imgs = {}
    ms_raws = {}
    for k, v in img_files.items():
        raw = mpimg.imread(str(v))
        if 'ms' in k:
            ms_raws[k] = raw
        else:
            imgs[k] = _crop_paraview_img(raw, crop_right_frac=0.0,
                                          replace_bg_white=True)

    if ms_raws:
        max_h = max(r.shape[0] for r in ms_raws.values())
        max_w = max(r.shape[1] for r in ms_raws.values())
        target_h, target_w = int(max_h * 0.92), int(max_w * 0.92)
        for k, raw in ms_raws.items():
            is_float = raw.max() <= 1.0
            arr = (raw * 255).astype(np.uint8) if is_float else raw
            pil = PILImage.fromarray(arr)
            pil.thumbnail((target_w, target_h), PILImage.LANCZOS)
            scaled = np.array(pil)
            canvas = np.full((max_h, max_w, arr.shape[2]), 255, dtype=np.uint8)
            ph = (max_h - scaled.shape[0]) // 2
            pw = (max_w - scaled.shape[1]) // 2
            canvas[ph:ph+scaled.shape[0], pw:pw+scaled.shape[1]] = scaled
            imgs[k] = canvas.astype(np.float32) / 255.0 if is_float else canvas

    # Flip both MS images so (c)↔(e) and (d)↔(f) orientations match
    imgs['clean_ms'] = np.fliplr(imgs['clean_ms']).copy()
    imgs['noisy_ms'] = np.fliplr(imgs['noisy_ms']).copy()

    # ── Load data ────────────────────────────────────────────────────────────
    pi = pd.read_csv(couplings_dir / 'mscoot_pi.csv', index_col=0).values
    xi = pd.read_csv(couplings_dir / 'mscoot_xi.csv', index_col=0).values
    clean_cp = pd.read_csv(ttk_dir / 'clean_critical_points.csv')
    noisy_cp = pd.read_csv(ttk_dir / 'noisy_critical_points.csv')
    clean_hyper = pd.read_csv(hyper_dir / 'hypergraph_clean.csv')
    noisy_hyper = pd.read_csv(hyper_dir / 'hypergraph_noisy.csv')
    n_clean_r, n_noisy_r = len(clean_hyper), len(noisy_hyper)

    sep_clean = pd.read_csv(ttk_dir / 'clean_separatrices_cells.csv')
    sep_noisy = pd.read_csv(ttk_dir / 'noisy_separatrices_cells.csv')

    # ── CP correspondence colors (pi) — MCOpt-style color transfer ───────────
    n_clean_cp = len(clean_cp)
    n_noisy_cp = len(noisy_cp)

    # Extract graph edges
    edges_clean = _extract_graph_edges(clean_cp, sep_clean)
    edges_noisy = _extract_graph_edges(noisy_cp, sep_noisy)

    # Sequential coloring by spatial position — encodes WHERE each CP is,
    # so color transfer in (g)→(h) reveals spatial coherence of the match.
    # Map diagonal position (x+y) to viridis (perceptually uniform,
    # colorblind-safe). Nearby CPs get similar colors by design.
    clean_xy = np.array([(clean_cp.iloc[i]['Points_0'],
                          clean_cp.iloc[i]['Points_1'])
                         for i in range(n_clean_cp)])
    diag = clean_xy[:, 0] + clean_xy[:, 1]
    diag_norm = (diag - diag.min()) / (diag.max() - diag.min() + 1e-12)
    cmap_seq = plt.cm.viridis
    clean_cp_colors = [cmap_seq(v)[:3] for v in diag_norm]

    # Color transfer: each noisy CP inherits the color of its best-matched
    # clean CP via the pi coupling matrix
    noisy_best_pi = np.argmax(pi, axis=0)
    noisy_cp_colors = [clean_cp_colors[noisy_best_pi[j]]
                       for j in range(n_noisy_cp)]

    # ── Region colors (xi) ───────────────────────────────────────────────────
    SPURIOUS_THRESHOLD = 0.002
    OPACITY_MIN, OPACITY_MAX = 0.30, 0.92

    clean_adj = build_region_adjacency(clean_hyper)
    clean_colors = graph_coloring(n_clean_r, clean_adj, REGION_PALETTE)

    noisy_best_match = np.argmax(xi, axis=0)
    noisy_max_coupling = np.max(xi, axis=0)
    max_xi = xi.max()

    noisy_colors, noisy_opacity, noisy_label_map = [], {}, {}
    for j in range(n_noisy_r):
        best_i = noisy_best_match[j]
        coupling = noisy_max_coupling[j]
        if coupling < SPURIOUS_THRESHOLD:
            noisy_colors.append((0.6, 0.6, 0.6, 0.4))
            noisy_opacity[j] = 0.0
            noisy_label_map[j] = ''
        else:
            noisy_colors.append(clean_colors[best_i])
            norm_c = coupling / max_xi
            noisy_opacity[j] = OPACITY_MIN + (OPACITY_MAX - OPACITY_MIN) * norm_c
            noisy_label_map[j] = str(best_i + 1)

    # Deduplicate labels
    label_groups = defaultdict(list)
    for j, lbl in noisy_label_map.items():
        if lbl:
            label_groups[lbl].append(j)
    for lbl, indices in label_groups.items():
        if len(indices) > 1:
            best_j = max(indices, key=lambda j: noisy_max_coupling[j])
            for j in indices:
                if j != best_j:
                    noisy_label_map[j] = ''

    # ── 2x4 layout ──────────────────────────────────────────────────────────
    TITLE_SIZE = 8.0
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.50))
    gs = GridSpec(2, 4, figure=fig, hspace=0.16, wspace=0.01,
                  left=0.005, right=0.995, top=0.94, bottom=0.02)

    # ── Column 0: Scalar fields ──
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(imgs['clean_3d'])
    ax_a.set_title('(a) Clean scalar field', fontweight='bold',
                   fontsize=TITLE_SIZE, pad=2)
    ax_a.axis('off')

    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.imshow(imgs['noisy_3d'])
    ax_b.set_title('(b) Noisy scalar field', fontweight='bold',
                   fontsize=TITLE_SIZE, pad=2)
    ax_b.axis('off')

    # ── Column 1: MS complexes (TTK) ──
    ax_c = fig.add_subplot(gs[0, 1])
    ax_c.imshow(imgs['clean_ms'])
    ax_c.set_title('(c) Clean MS complex', fontweight='bold',
                   fontsize=TITLE_SIZE, pad=2)
    ax_c.axis('off')

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.imshow(imgs['noisy_ms'])
    ax_d.set_title('(d) MS complex (noisy field)', fontweight='bold',
                   fontsize=TITLE_SIZE, pad=2)
    ax_d.axis('off')

    # ── Column 2: Region matching (xi) ──
    _edge_kw_clean = dict(edge_linewidth=1.2,
                          edge_color_uniform=(0.25, 0.25, 0.25, 0.65))
    _edge_kw_noisy = dict(edge_linewidth=0.9,
                          edge_color_uniform=(0.25, 0.25, 0.25, 0.55))
    FILL_ALPHA = 0.40

    ax_e = fig.add_subplot(gs[0, 2])
    _draw_xi_correspondence_panel(
        ax_e, clean_cp, clean_hyper, clean_colors,
        fill_regions=True, fill_alpha=FILL_ALPHA,
        show_labels=True, label_fontsize=5.0, **_edge_kw_clean)
    ax_e.set_title(f'(e) Clean regions ($\\xi$, m={n_clean_r})',
                   fontweight='bold', fontsize=TITLE_SIZE, pad=2)

    ax_f = fig.add_subplot(gs[1, 2])
    _draw_xi_correspondence_panel(
        ax_f, noisy_cp, noisy_hyper, noisy_colors,
        opacity_map=noisy_opacity, label_map=noisy_label_map,
        label_nudges={'26': (0, 5)},
        fill_regions=True, fill_alpha=FILL_ALPHA,
        show_labels=True, label_fontsize=4.0, min_label_area=25,
        **_edge_kw_noisy)
    ax_f.set_title(f'(f) Noisy-field regions ($\\xi$, m={n_noisy_r})',
                   fontweight='bold', fontsize=TITLE_SIZE, pad=2)

    # ── Highlight split separatrices in panel (f) ────────────────────────────
    split_groups = defaultdict(list)
    for j in range(n_noisy_r):
        if noisy_max_coupling[j] >= SPURIOUS_THRESHOLD:
            split_groups[noisy_best_match[j]].append(j)

    for clean_id, indices in split_groups.items():
        if len(indices) <= 1:
            continue
        for ai in range(len(indices)):
            for bi in range(ai + 1, len(indices)):
                a, b = indices[ai], indices[bi]
                row_a, row_b = noisy_hyper.iloc[a], noisy_hyper.iloc[b]
                cps_a = ({int(row_a['min_id']), int(row_a['max_id'])}
                         | {int(s) for s in eval(row_a['boundary_saddles'])})
                cps_b = ({int(row_b['min_id']), int(row_b['max_id'])}
                         | {int(s) for s in eval(row_b['boundary_saddles'])})
                shared = sorted(cps_a & cps_b)
                if len(shared) >= 2:
                    coords = np.array([[noisy_cp.iloc[c]['Points_0'],
                                        noisy_cp.iloc[c]['Points_1']]
                                       for c in shared])
                    ax_f.plot(coords[:, 0], coords[:, 1],
                              color='black', linewidth=1.8, alpha=0.7,
                              zorder=4, solid_capstyle='round')

    # ── Column 3: CP correspondence (pi) — color transfer ──
    ax_g = fig.add_subplot(gs[0, 3])
    _draw_cp_correspondence_panel(ax_g, clean_cp, edges_clean, clean_cp_colors,
                                   edge_lw=2.0, edge_alpha=0.82)
    ax_g.set_title(f'(g) Clean CPs ($\\pi$, n={n_clean_cp})', fontweight='bold',
                   fontsize=TITLE_SIZE, pad=2)

    ax_h = fig.add_subplot(gs[1, 3])
    _draw_cp_correspondence_panel(ax_h, noisy_cp, edges_noisy, noisy_cp_colors,
                                   edge_lw=2.0, edge_alpha=0.82)
    ax_h.set_title(f'(h) Noisy-field CPs ($\\pi$, n={n_noisy_cp})', fontweight='bold',
                   fontsize=TITLE_SIZE, pad=2)

    # ── Annotation between (e)/(f): xi ───────────────────────────────────────
    pos_e = ax_e.get_position()
    pos_f = ax_f.get_position()
    ann_x_xi = (pos_e.x0 + pos_e.x1) / 2
    ann_y_xi = pos_f.y1 + 0.88 * (pos_e.y0 - pos_f.y1)
    fig.text(ann_x_xi, ann_y_xi,
             r'shared ID + color $=\;$ $\boldsymbol{\xi}$ region correspondence',
             ha='center', va='center',
             fontsize=6.5, fontweight='bold', color='0.25',
             bbox=dict(facecolor='white', edgecolor='0.50', lw=0.4,
                       pad=2, boxstyle='round,pad=0.3'))

    # ── Annotation between (g)/(h): pi ───────────────────────────────────────
    pos_g = ax_g.get_position()
    pos_h = ax_h.get_position()
    ann_x_pi = (pos_g.x0 + pos_g.x1) / 2
    ann_y_pi = pos_h.y1 + 0.88 * (pos_g.y0 - pos_h.y1)
    fig.text(ann_x_pi, ann_y_pi,
             r'shared color $=\;$ $\boldsymbol{\pi}$ CP correspondence',
             ha='center', va='center',
             fontsize=6.5, fontweight='bold', color='0.25',
             bbox=dict(facecolor='white', edgecolor='0.50', lw=0.4,
                       pad=2, boxstyle='round,pad=0.3'))

    # ── Pipeline arrows between columns (line + arrowhead) ──────────────────
    from matplotlib.patches import FancyArrowPatch
    for ax_left, ax_right in [(ax_a, ax_c), (ax_b, ax_d),
                               (ax_c, ax_e), (ax_d, ax_f)]:
        pos_l = ax_left.get_position()
        pos_r = ax_right.get_position()
        gap = pos_r.x0 - pos_l.x1
        cy = (pos_l.y0 + pos_l.y1) / 2
        x0 = pos_l.x1 + gap * 0.15
        x1 = pos_r.x0 - gap * 0.15
        arrow = FancyArrowPatch(
            (x0, cy), (x1, cy),
            arrowstyle='-|>', mutation_scale=7,
            linewidth=0.8, color='0.40',
            transform=fig.transFigure, clip_on=False)
        fig.patches.append(arrow)

    # ── "&" between columns 2 and 3 (joint outputs of MS-COOT) ──────────
    for ax_left, ax_right in [(ax_e, ax_g), (ax_f, ax_h)]:
        pos_l = ax_left.get_position()
        pos_r = ax_right.get_position()
        cx = (pos_l.x1 + pos_r.x0) / 2
        cy = (pos_l.y0 + pos_l.y1) / 2
        fig.text(cx, cy, '&', ha='center', va='center',
                 fontsize=10, fontweight='bold', color='0.40',
                 transform=fig.transFigure)

    # ── CP type legend (between rows, under MS complex column) ───────────────
    legend_handles = [
        Line2D([0], [0], marker=CP_MARKERS[0], color='none',
                markerfacecolor=CP_COLORS_LEGEND[0], markeredgecolor='black',
                markeredgewidth=0.3, markersize=5, label=CP_LABELS[0]),
        Line2D([0], [0], marker=CP_MARKERS[1], color='none',
                markerfacecolor=CP_COLORS_LEGEND[1], markeredgecolor='black',
                markeredgewidth=0.8, markersize=5, label=CP_LABELS[1]),
        Line2D([0], [0], marker=CP_MARKERS[2], color='none',
                markerfacecolor=CP_COLORS_LEGEND[2], markeredgecolor='black',
                markeredgewidth=0.3, markersize=5, label=CP_LABELS[2]),
    ]
    pos_c = ax_c.get_position()
    pos_d = ax_d.get_position()
    leg_x = (pos_c.x0 + pos_c.x1) / 2
    leg_y = pos_d.y1 + 0.82 * (pos_c.y0 - pos_d.y1)
    fig.legend(handles=legend_handles, loc='center', ncol=3,
               fontsize=6.0, frameon=True, fancybox=False, edgecolor='0.7',
               handletextpad=0.3, columnspacing=0.8,
               bbox_to_anchor=(leg_x, leg_y))

    # ── Save ─────────────────────────────────────────────────────────────────
    out_base = fig_dir / 'fig1_method_overview_v3'
    fig.savefig(f'{out_base}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{out_base}.png', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'Saved {out_base}.pdf/.png')


if __name__ == '__main__':
    main()
