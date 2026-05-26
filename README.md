# MS-COOT: Comparing Morse-Smale Complexes with Co-Optimal Transport

This repository implements **MS-COOT**, a method for comparing Morse-Smale (MS) complexes using co-optimal transport. MS-COOT represents each MS complex as a **measure hypernetwork** and jointly optimizes two transport couplings: one matching critical points ($\pi$) and one matching regions ($\xi$). The region coupling $\xi$ is the key contribution — it enables explicit region-to-region correspondence that detects structural events such as splitting and merging, which are invisible to graph-based baselines.

## Methods

| | **WD** | **GWD** | **FGW** | **MS-COOT** |
|---|---|---|---|---|
| **Node measure** | Uniform | Uniform | Uniform | Persistence-based |
| **Spatial cost** | L2 / diagonal | — | L2 / diagonal | Type mismatch penalty |
| **Structure cost** | — | Shortest-path / max | Shortest-path / max | Hypernetwork function $\omega$ |
| **Solver** | `ot.emd` | `ot.gromov_wasserstein` | `ot.fused_gromov_wasserstein` | `ot.coot.co_optimal_transport` |
| **Output** | CP coupling $\pi$ | CP coupling $\pi$ | CP coupling $\pi$ | CP coupling $\pi$ **+ region coupling $\xi$** |

## Installation

```bash
git clone https://github.com/gm3g11/HyperCOT-MS.git
cd HyperCOT-MS
pip install -e .
```

**Dependencies:** numpy, pandas, scipy, matplotlib, POT, networkx, gudhi, PyYAML, tqdm

**Optional (for TTK extraction):**
```bash
pip install -e ".[ttk]"
```

## Quick Start (Sinusoidal Example)

The repository includes a sinusoidal dataset (clean vs. noisy surfaces) with pre-extracted TTK output for immediate testing:

```bash
# 1. Compute MS-COOT components (hypergraphs, mu, nu, omega)
python scripts/compute_components.py --dataset sinusoidal --sigma 0.3

# 2. Compute consecutive pair distances
python scripts/compute_consecutive.py --dataset sinusoidal --methods wd,gwd,fgw,mscoot
```

## Full Pipeline

For temporal datasets with multiple timesteps:

```bash
# Step 1: Compute hypergraphs and MS-COOT components
python scripts/compute_components.py --dataset vortexstreet --sigma 0.3

# Step 2: Compute all-pairs distance matrices
python scripts/compute_distances.py --dataset vortexstreet --methods wd,gwd,fgw,mscoot

# Step 3: Classification (TOSCA / Viscous Finger)
python scripts/classify_tosca.py --dataset tosca
python scripts/classify_ensemble.py

# Step 4: Generate paper figures
python scripts/fig_heatedcylinder_phase.py
python scripts/fig_ionfront_combined.py
python scripts/fig_tosca_ablation_sweeps.py
```

## Project Structure

```
├── src/mscoot/                 # Python package
│   ├── config.py               # Dataset configuration (dataset.yaml)
│   ├── data_adapter.py         # Unified data loading
│   ├── utils.py                # Logging, validation, metrics
│   ├── graph/                  # Adjacency, hypergraph, virtual center
│   ├── measures/               # Persistence, node measure (mu), edge measure (nu), omega
│   ├── solvers/                # WD, GWD, FGW, MS-COOT
│   ├── ttk/                    # TTK extraction (optional)
│   └── visualization/          # Publication-quality plotting
├── scripts/                    # CLI entry points and figure generation
├── datasets/                   # Dataset configs + sinusoidal example
│   ├── sinusoidal/             # Example: TTK output included
│   ├── vortexstreet/           # Config only (dataset.yaml)
│   ├── ionization_front/       # Config only
│   ├── heatedcylinder-800-899/ # Config only
│   ├── tosca/                  # Config only
│   └── viscous_finger/         # Config only
└── pyproject.toml              # Package metadata
```

## Datasets

| Dataset | Type | Timesteps | Description |
|---------|------|-----------|-------------|
| **Vortex Street** | 2D time-varying | 157 | Karman vortex shedding — periodicity detection |
| **Ionization Front** | 2D time-varying | 123 | Gas dynamics — region merge detection |
| **Heated Cylinder** | 2D time-varying | 201 | Flow around obstacle — phase transition |
| **TOSCA** | 3D surface meshes | — | Non-rigid shape classification (9 categories) |
| **Viscous Finger** | 3D volumetric | ~50-80 | Ensemble classification by resolution |

Datasets other than sinusoidal require separate download. TTK output is generated via [ParaView](https://www.paraview.org/) with the [TTK plugin](https://topology-tool-kit.github.io/).

## Paper Figures

| Script | Figure |
|--------|--------|
| `scripts/fig_vf_teaser.py` | Viscous Finger teaser |
| `scripts/generate_fig1_v3.py` | Method overview |
| `scripts/generate_composite_vs_panels.py` | Vortex Street agreement |
| `scripts/fig_ionfront_combined.py` | Ionization Front |
| `scripts/fig_heatedcylinder_phase.py` | Heated Cylinder phase transition |
| `scripts/fig_tosca_ablation_sweeps.py` | TOSCA parameter sensitivity |

## Citation

```bibtex
@article{mscoot2026,
  title     = {{MS-COOT}: Comparing {Morse-Smale} Complexes with Co-Optimal Transport},
  author    = {Meng, Guangyu and Wang, Bei},
  journal   = {IEEE Transactions on Visualization and Computer Graphics},
  year      = {2026}
}
```

## License

MIT License
