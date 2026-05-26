#!/usr/bin/env python3
"""TOSCA ablation sweeps: 4-panel figure (Fig. 10).

Data from ablation_coot_sweep.py runs. Accuracy values are k-NN LOOCV on
80 TOSCA meshes (9 classes).

Usage:
    python scripts/fig_tosca_ablation_sweeps.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_TOSCA = ROOT / 'results' / 'tosca' / 'figures'
OUT_PUB = ROOT / 'results' / 'publication_figures'

# ── Ablation data (from ablation_coot_sweep.py stdout) ─────────────

# Sample cost type (alpha=0.5, eps=0, self-norm, bcd=100)
COST_LABELS = ['Type\n(0/1)', 'Scalar\n(AGD)', 'Type +\nScalar']
COST_K1 = [65.0, 86.2, 83.8]
COST_K3 = [57.5, 86.2, 82.5]
COST_K5 = [57.5, 82.5, 81.2]

# Alpha sweep (scalar M_samp, eps=0, self-norm, bcd=100)
ALPHA_VALS = [0.1, 0.3, 0.5, 0.7, 0.9]
ALPHA_K1 = [81.2, 86.2, 86.2, 86.2, 85.0]
ALPHA_K3 = [83.8, 85.0, 86.2, 86.2, 86.2]
ALPHA_K5 = [77.5, 82.5, 82.5, 83.8, 82.5]

# Sigma sweep (persistence image bandwidth)
SIGMA_VALS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
SIGMA_K1 = [82.5, 82.5, 86.2, 83.8, 83.8, 82.5]
SIGMA_K3 = [80.0, 81.2, 86.2, 82.5, 81.2, 81.2]
SIGMA_K5 = [77.5, 78.8, 82.5, 82.5, 80.0, 80.0]

# Epsilon sweep (scalar M_samp, alpha=0.5, self-norm, bcd=100)
EPS_LABELS = ['0 (EMD)', '0.001', '0.01', '0.1']
EPS_K1 = [86.2, 80.0, 86.2, 76.2]
EPS_K3 = [86.2, 80.0, 82.5, 76.2]
EPS_K5 = [82.5, 80.0, 81.2, 75.0]

# ── Color palette (colorblind-safe: blue, orange, gray) ───────────
C1 = '#2166ac'  # k=1 dark blue
C3 = '#e66101'  # k=3 orange
C5 = '#969696'  # k=5 gray


def setup_style():
    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif',
        'text.usetex': False,
        'font.size': 8,
        'axes.titlesize': 8.5,
        'axes.labelsize': 7.5,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'legend.fontsize': 6,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })



def _style_ax(ax, ylabel=True):
    """Common axis styling: grid, spines, y-label."""
    ax.yaxis.grid(True, linewidth=0.25, alpha=0.5, color='#bbbbbb', zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([50, 60, 70, 80, 90])
    ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9'])
    ax.set_ylim(50, 90)
    if ylabel:
        ax.set_ylabel('Recall')
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)


def _add_legend(ax, loc='lower right'):
    ax.legend(loc=loc, frameon=True, fancybox=False,
              edgecolor='#cccccc', framealpha=0.95,
              handlelength=1.4, handletextpad=0.4,
              borderpad=0.25, labelspacing=0.2, fontsize=5.5)


def _line_panel(ax, xvals, k1, k3, k5, xlabel_right, title,
                default_x=None, legend_loc='lower right', ylabel=True):
    """Line plot panel with x-label placed to the right of the axis."""
    ax.plot(xvals, k1, 'o-', color=C1, markersize=2.5, linewidth=1.0,
            markeredgewidth=0, label=r'$k{=}1$', zorder=3)
    ax.plot(xvals, k3, 's--', color=C3, markersize=2.3, linewidth=0.85,
            markeredgewidth=0, label=r'$k{=}3$', zorder=3)
    ax.plot(xvals, k5, '^:', color=C5, markersize=2.3, linewidth=0.85,
            markeredgewidth=0.3, markeredgecolor=C5, label=r'$k{=}5$', zorder=3)
    if default_x is not None:
        ax.axvline(default_x, color='#888888', linestyle='--',
                   linewidth=0.5, zorder=1)
    # Place x-label at right end of axis instead of below center
    ax.set_xlabel('')
    ax.annotate(xlabel_right, xy=(1.0, -0.02), xycoords='axes fraction',
                ha='left', va='top', fontsize=7,
                xytext=(3, -1), textcoords='offset points')
    ax.set_title(title, pad=2)
    _style_ax(ax, ylabel=ylabel)
    _add_legend(ax, legend_loc)


def _bar_panel(ax, labels, k1, k3, k5, title, legend_loc='upper left',
               ylabel=True):
    """Grouped bar panel."""
    x = np.arange(len(labels))
    w = 0.22
    ax.bar(x - w, k1, w, color=C1, label=r'$k{=}1$',
           edgecolor='white', linewidth=0.3, zorder=2)
    ax.bar(x,     k3, w, color=C3, label=r'$k{=}3$',
           edgecolor='white', linewidth=0.3, zorder=2)
    ax.bar(x + w, k5, w, color=C5, label=r'$k{=}5$',
           edgecolor='white', linewidth=0.3, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title, pad=2)
    _style_ax(ax, ylabel=ylabel)
    _add_legend(ax, legend_loc)


def main():
    setup_style()

    # Use gridspec for unequal row heights: bar rows taller, line rows shorter
    fig = plt.figure(figsize=(3.5, 2.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82],
                          hspace=0.50, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # (a) Sample cost C — bar chart
    _bar_panel(ax_a, COST_LABELS, COST_K1, COST_K3, COST_K5,
               r'(a) Sample cost $C$', legend_loc='upper left', ylabel=True)

    # (b) Alpha — line chart, x-label at right
    _line_panel(ax_b, ALPHA_VALS, ALPHA_K1, ALPHA_K3, ALPHA_K5,
                r'$\alpha$',
                r'(b) $\alpha$ (balance weight)',
                default_x=0.5, legend_loc='lower right', ylabel=True)
    ax_b.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax_b.set_xlim(0.0, 1.0)

    # (c) Sigma — line chart, x-label at right
    _line_panel(ax_c, SIGMA_VALS, SIGMA_K1, SIGMA_K3, SIGMA_K5,
                r'$\sigma$',
                r'(c) $\sigma$ (PI bandwidth)',
                default_x=0.3, legend_loc='lower right', ylabel=True)
    ax_c.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_c.set_xlim(0.0, 1.1)
    ax_c.set_title(ax_c.get_title(), pad=6)

    # (d) Epsilon — bar chart, legend lower right
    _bar_panel(ax_d, EPS_LABELS, EPS_K1, EPS_K3, EPS_K5,
               r'(d) $\varepsilon$ (Sinkhorn regularization)',
               legend_loc='lower right', ylabel=True)
    ax_d.set_title(ax_d.get_title(), pad=6)

    OUT_TOSCA.mkdir(parents=True, exist_ok=True)
    OUT_PUB.mkdir(parents=True, exist_ok=True)

    for d, name in [(OUT_TOSCA, 'tosca_ablation_sweeps'),
                    (OUT_PUB, 'fig_tosca_ablation_sweeps')]:
        fig.savefig(d / f'{name}.pdf')
        fig.savefig(d / f'{name}.png')

    print(f'Saved to {OUT_TOSCA} and {OUT_PUB}')


if __name__ == '__main__':
    main()
