"""Shared visualization style constants and rcParams."""

import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams

# Publication-quality defaults
RCPARAMS = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
}

# Method colors (ColorBrewer Set1)
COLOR_WD = '#377eb8'
COLOR_GWD = '#4daf4a'
COLOR_FGW = '#ff7f00'
COLOR_COOT = '#e41a1c'

METHOD_COLORS = {
    'WD': COLOR_WD,
    'GWD': COLOR_GWD,
    'FGW': COLOR_FGW,
    'HyperCOT': COLOR_COOT,
}

# CP type colors
CP_COLORS = {0: '#2166ac', 1: '#b2182b', 2: '#762a83'}  # min=blue, saddle=red, max=purple
CP_LABELS = {0: 'Minimum', 1: 'Saddle', 2: 'Maximum'}


def apply_style():
    """Apply publication rcParams globally."""
    rcParams.update(RCPARAMS)
