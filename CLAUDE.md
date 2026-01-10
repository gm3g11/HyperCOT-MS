# Project: MS Complex Comparison using Optimal Transport

## Quick Context
This project compares Morse-Smale complexes from clean vs noisy sinusoidal surfaces using:
1. Wasserstein Distance (WD) - scalar field values only (feature-based)
2. Gromov-Wasserstein Distance (GWD) - graph structure (structure-based)
3. HyperCOT - combined node features + hypergraph structure

## Current Status
- ✅ WD implementation completed (uses `data` column = scalar field values)
- ✅ GWD implementation completed with Dijkstra + Euclidean edge weights
- ✅ HyperCOT implementation completed
- ✅ All visualizations completed in consistent format

## Method Comparison

| Method | Feature | Structure | Output |
|--------|---------|-----------|--------|
| WD | Scalar field value (`data`) | None | CP coupling |
| GWD | None | Graph distances (Dijkstra) | CP coupling |
| HyperCOT | μ (persistence-based) | ω (hypergraph + VC) | CP coupling (π) + Region coupling (ξ) |

## Results Summary

| Method | Distance | Type Preservation |
|--------|----------|-------------------|
| WD | 0.0402 | 90.0% |
| GWD | 0.0037 | 72.5% |

## Data Summary

### Critical Points (after persistence filtration)
| Dataset | Total CPs | Minima | Saddles | Maxima |
|---------|-----------|--------|---------|--------|
| Clean   | 49        | 16     | 24      | 9      |
| Noisy   | 65        | 22     | 32      | 11     |

### Hypergraph Structure
| Dataset | Regions | Hyperedge Size |
|---------|---------|----------------|
| Clean   | 36      | 4 (1 min + 2 sad + 1 max) |
| Noisy   | 47      | 4 (1 min + 2 sad + 1 max) |

## Key Files

### Data Files (Input)
- `clean_critical_points.csv` / `noisy_critical_points.csv` - CP coordinates and types
  - `data` column = scalar field value (used by WD)
  - `Points_0`, `Points_1`, `Points_2` = x, y, z coordinates
- `clean_separatrices_cells.csv` / `noisy_separatrices_cells.csv` - Edge connectivity
- `clean_segmentation.csv` / `noisy_segmentation.csv` - Region segmentation
- `hypergraph_clean.csv` / `hypergraph_noisy.csv` - Hyperedge definitions

### HyperCOT Intermediate Data
- `clean_node_measure.csv` / `noisy_node_measure.csv` - μ values per CP
- `clean_nu.csv` / `noisy_nu.csv` - ν values per hyperedge
- `clean_omega.csv` / `noisy_omega.csv` - ω distance matrices
- `clean_virtual_centers.csv` / `noisy_virtual_centers.csv` - Virtual center coordinates
- `hypercot_pi.csv` - π matrix (node coupling)
- `hypercot_xi.csv` - ξ matrix (hyperedge coupling)

### Scripts
- `ms_complex_comparison.py` - Main comparison algorithms (WD, GWD)
- `visualize_wd_correspondence.py` - WD visualization (coupling matrix + bipartite)
- `visualize_gwd_correspondence.py` - GWD visualization (coupling + bipartite + edge preservation)
- `compute_node_measure.py` - Compute μ using extended persistence
- `compute_nu.py` - Compute ν (hyperedge measure)
- `compute_omega.py` - Compute ω (hypernetwork function with augmented graph)
- `compute_hypercot.py` - HyperCOT optimization (alternating Sinkhorn)
- `visualize_hypercot_final.py` - HyperCOT visualization
- `generate_hypergraph.py` - Generate hypergraph CSVs
- `visualize_vc_adjacency.py` - VC adjacency visualization (3-panel: VC generation, augmented graph, shortest path)

### Output Figures
- `wd_correspondence_refined.png` - WD results (coupling matrix + CP correspondence)
- `gwd_point_edge_correspondence.png` - GWD results (coupling + correspondence + edge preservation)
- `hypercot_detailed_correspondence.png` - HyperCOT results (spatial + π + ξ)
- `clean_vc_adjacency.png` / `noisy_vc_adjacency.png` - VC adjacency (VC generation + augmented graph + shortest path)

## HyperCOT Method Summary

### Node Measure (μ)
- Extended persistence diagram using z-values as filtration
- Persistence image representation with Gaussian kernel
- C(v) = Σ_k I_f[k] · φ_{(b_v,p_v)}(c_k), normalized

### Hyperedge Measure (ν)
- ν(e) = Σ_{v ∈ boundary(e)} μ(v), normalized

### Hypernetwork Function (ω)
- Augmented graph: CP↔CP (separatrices) + CP↔VC (boundary) + VC↔VC (adjacent regions)
- Virtual center (VC): Intersection of (min,max) and (saddle,saddle) lines
- **VC-VC adjacency rule**: Two regions are adjacent if they share **2+ boundary CPs**
  - 2+ shared CPs = regions share a separatrix (edge) → 4-neighbor connectivity
  - 1 shared CP = regions share only a vertex (corner) → excluded (would be 8-neighbor)
- Shortest path using Dijkstra with L2 edge weights

### Optimization
- Alternating Sinkhorn with entropy regularization (reg=0.005)
- Outputs: π (49×65 node coupling), ξ (36×47 hyperedge coupling)

## TTK Data Format Notes
- `Points_0`, `Points_1`, `Points_2` = (x, y, z) spatial coordinates
- `data` = scalar field value at CP
- `CellDimension` = CP type (0=min, 1=saddle, 2=max)
- `DescendingManifold` = Point ID of minimum (direct)
- `AscendingManifold` + offset = Point ID of maximum
- `SeparatrixType`: 0 = descending (saddle→min), 1 = ascending (saddle→max)
