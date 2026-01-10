"""
Generate Hypergraph CSVs and COOT Input Visualization
Creates hypergraph_clean.csv, hypergraph_noisy.csv, and coot_input.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
from pathlib import Path

BASE_PATH = Path(__file__).parent.resolve()

# Colors for CP types
CP_COLORS = {
    0: '#2166ac',  # min - blue
    1: '#4daf4a',  # saddle - green
    2: '#e41a1c'   # max - red
}

CP_MARKERS = {
    0: 'o',  # min - circle
    1: 's',  # saddle - square
    2: '^'   # max - triangle
}


def load_data(prefix):
    """Load critical points, segmentation, and separatrices data."""
    cp_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_critical_points.csv"))
    seg_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_segmentation.csv"))
    sep_df = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_separatrices_cells.csv"))

    # Standardize critical points DataFrame
    cp = pd.DataFrame({
        'point_id': cp_df['Point ID'],
        'cell_id': cp_df['CellId'],
        'cp_type': cp_df['CellDimension'],
        'x': cp_df['Points_0'],
        'y': cp_df['Points_1'],
        'z': cp_df['Points_2']
    })

    return cp, seg_df, sep_df


def build_hypergraph(cp_df, seg_df, sep_df):
    """
    Build hypergraph from MS complex data.

    Each region (hyperedge) contains:
    - 1 minimum
    - 1 maximum
    - boundary saddles (connected to both min and max via separatrices)

    Returns:
        H: Incidence matrix (n_cp x n_regions)
        region_info: DataFrame with region metadata
    """
    n_cp = len(cp_df)

    # Get CP info by type
    cp_types = cp_df['cp_type'].values
    cp_point_ids = cp_df['point_id'].values
    cp_cell_ids = cp_df['cell_id'].values

    minima_idx = np.where(cp_types == 0)[0]
    saddle_idx = np.where(cp_types == 1)[0]
    maxima_idx = np.where(cp_types == 2)[0]

    # Build mappings
    point_id_to_idx = {pid: idx for idx, pid in enumerate(cp_point_ids)}
    cell_id_to_idx = {cid: idx for idx, cid in enumerate(cp_cell_ids)}

    # Get max offset (for AscendingManifold -> max Point ID mapping)
    max_point_ids = cp_point_ids[maxima_idx]
    max_offset = min(max_point_ids) if len(max_point_ids) > 0 else 0

    # Build saddle connectivity from separatrices
    # SeparatrixType: 0 = descending (saddle -> min), 1 = ascending (saddle -> max)
    unique_sep = sep_df.drop_duplicates(subset=['SeparatrixId'])[
        ['SeparatrixId', 'SourceId', 'DestinationId', 'SeparatrixType']
    ]

    # Map CellId to CP index for saddles
    saddle_cell_ids = set(cp_cell_ids[saddle_idx])

    # Build saddle -> {connected mins/maxs} using CellId
    saddle_to_mins = {}  # saddle_cell_id -> set of min_cell_ids
    saddle_to_maxs = {}  # saddle_cell_id -> set of max_cell_ids

    for _, row in unique_sep.iterrows():
        src_cell = int(row['SourceId'])
        dst_cell = int(row['DestinationId'])
        sep_type = int(row['SeparatrixType'])

        if src_cell in saddle_cell_ids:
            if sep_type == 0:  # Descending: saddle -> min
                if src_cell not in saddle_to_mins:
                    saddle_to_mins[src_cell] = set()
                saddle_to_mins[src_cell].add(dst_cell)
            else:  # Ascending: saddle -> max
                if src_cell not in saddle_to_maxs:
                    saddle_to_maxs[src_cell] = set()
                saddle_to_maxs[src_cell].add(dst_cell)

    # Get unique regions from segmentation
    valid_seg = seg_df[(seg_df['DescendingManifold'] >= 0) &
                       (seg_df['AscendingManifold'] >= 0)].copy()

    region_groups = valid_seg.groupby('MorseSmaleManifold').agg({
        'DescendingManifold': 'first',
        'AscendingManifold': 'first',
        'Point ID': 'count'
    }).reset_index()
    region_groups.columns = ['ms_manifold', 'desc_manifold', 'asc_manifold', 'region_size']

    # Map to CP Point IDs
    # DescendingManifold = Point ID of minimum directly
    # AscendingManifold + max_offset = Point ID of maximum
    region_groups['min_point_id'] = region_groups['desc_manifold']
    region_groups['max_point_id'] = region_groups['asc_manifold'] + max_offset

    n_regions = len(region_groups)
    H = np.zeros((n_cp, n_regions))

    region_info_list = []

    for region_idx, row in region_groups.iterrows():
        min_pid = int(row['min_point_id'])
        max_pid = int(row['max_point_id'])
        region_size = int(row['region_size'])

        # Get CP indices
        min_cp_idx = point_id_to_idx.get(min_pid, -1)
        max_cp_idx = point_id_to_idx.get(max_pid, -1)

        if min_cp_idx < 0 or max_cp_idx < 0:
            continue

        # Get CellIds for this min and max
        min_cell_id = cp_cell_ids[min_cp_idx]
        max_cell_id = cp_cell_ids[max_cp_idx]

        # Add min and max to hyperedge
        H[min_cp_idx, region_idx] = 1
        H[max_cp_idx, region_idx] = 1

        # Find boundary saddles (connected to both this min AND this max)
        boundary_saddles = []
        for s_idx in saddle_idx:
            s_cell_id = cp_cell_ids[s_idx]
            mins_connected = saddle_to_mins.get(s_cell_id, set())
            maxs_connected = saddle_to_maxs.get(s_cell_id, set())

            if min_cell_id in mins_connected and max_cell_id in maxs_connected:
                H[s_idx, region_idx] = 1
                boundary_saddles.append(int(s_idx))

        # Build hyperedge list
        hyperedge = [int(min_cp_idx)] + sorted(boundary_saddles) + [int(max_cp_idx)]

        region_info_list.append({
            'region_id': region_idx + 1,
            'min_id': int(min_cp_idx),
            'max_id': int(max_cp_idx),
            'num_saddles': len(boundary_saddles),
            'boundary_saddles': str(boundary_saddles),
            'hyperedge': str(hyperedge),
            'hyperedge_size': len(hyperedge),
            'region_size': region_size
        })

    region_info = pd.DataFrame(region_info_list)

    return H, region_info


def plot_coot_input(cp1, cp2, H1, H2, region_info1, region_info2, save_path):
    """Create COOT input visualization showing hypergraph structures."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    plt.subplots_adjust(wspace=0.15)  # Reduce horizontal space between subplots

    def plot_hypergraph(ax, cp_df, H, region_info, title):
        x = cp_df['x'].values
        y = cp_df['y'].values
        types = cp_df['cp_type'].values
        n_cp = len(cp_df)
        n_regions = H.shape[1]

        # Generate colors for regions
        np.random.seed(42)
        region_colors = plt.cm.tab20(np.linspace(0, 1, 20))

        # Draw regions as polygons
        patches = []
        colors = []

        for region_idx in range(n_regions):
            cp_in_region = np.where(H[:, region_idx] > 0)[0]
            if len(cp_in_region) >= 3:
                # Get coordinates
                region_x = x[cp_in_region]
                region_y = y[cp_in_region]

                # Sort points by angle for convex hull-like polygon
                cx, cy = np.mean(region_x), np.mean(region_y)
                angles = np.arctan2(region_y - cy, region_x - cx)
                sorted_idx = np.argsort(angles)

                vertices = np.column_stack([region_x[sorted_idx], region_y[sorted_idx]])
                polygon = Polygon(vertices, closed=True)
                patches.append(polygon)
                colors.append(region_colors[region_idx % 20])

        # Add patches with transparency
        p = PatchCollection(patches, alpha=0.4)
        p.set_facecolors(colors)
        p.set_edgecolors('gray')
        p.set_linewidth(0.5)
        ax.add_collection(p)

        # Plot critical points
        for cp_type in [0, 1, 2]:
            mask = types == cp_type
            ax.scatter(x[mask], y[mask],
                      c=CP_COLORS[cp_type],
                      marker=CP_MARKERS[cp_type],
                      s=80, zorder=5,
                      edgecolors='black', linewidths=0.5)

        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_aspect('equal')


    # Plot both hypergraphs
    n_cp1, n_regions1 = len(cp1), H1.shape[1]
    n_cp2, n_regions2 = len(cp2), H2.shape[1]

    plot_hypergraph(ax1, cp1, H1, region_info1,
                   f'Clean Surface Hypergraph\n({n_cp1} CPs, {n_regions1} regions)')
    plot_hypergraph(ax2, cp2, H2, region_info2,
                   f'Noisy Surface Hypergraph\n({n_cp2} CPs, {n_regions2} regions)')

    # Shared legend at the bottom center (between plots and footer)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CP_COLORS[0],
                  markersize=10, label='Minimum', markeredgecolor='black', markeredgewidth=0.5),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=CP_COLORS[1],
                  markersize=10, label='Saddle', markeredgecolor='black', markeredgewidth=0.5),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=CP_COLORS[2],
                  markersize=10, label='Maximum', markeredgecolor='black', markeredgewidth=0.5),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
              framealpha=0.95, bbox_to_anchor=(0.5, 0.06))

    # Main title
    fig.suptitle('COOT Input: Hypergraph Structure (MS Complex Regions)',
                fontsize=15, fontweight='bold', y=0.98)

    # Footer
    fig.text(0.5, 0.02,
            'Each colored region is a hyperedge containing 1 min + 2 saddles + 1 max. '
            'COOT combines scalar values + hypergraph co-membership.',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("GENERATING HYPERGRAPH DATA")
    print("=" * 60)

    # Load data
    print("\nLoading clean data...")
    cp1, seg1, sep1 = load_data("clean")
    print(f"  CPs: {len(cp1)} (min={sum(cp1['cp_type']==0)}, sad={sum(cp1['cp_type']==1)}, max={sum(cp1['cp_type']==2)})")

    print("\nLoading noisy data...")
    cp2, seg2, sep2 = load_data("noisy")
    print(f"  CPs: {len(cp2)} (min={sum(cp2['cp_type']==0)}, sad={sum(cp2['cp_type']==1)}, max={sum(cp2['cp_type']==2)})")

    # Build hypergraphs
    print("\nBuilding clean hypergraph...")
    H1, region_info1 = build_hypergraph(cp1, seg1, sep1)
    print(f"  Regions: {H1.shape[1]}")
    print(f"  Incidence matrix: {H1.shape}")

    print("\nBuilding noisy hypergraph...")
    H2, region_info2 = build_hypergraph(cp2, seg2, sep2)
    print(f"  Regions: {H2.shape[1]}")
    print(f"  Incidence matrix: {H2.shape}")

    # Save CSVs
    print("\n" + "=" * 60)
    print("SAVING HYPERGRAPH CSVs")
    print("=" * 60)

    csv_path1 = os.path.join(BASE_PATH, "hypergraph_clean.csv")
    csv_path2 = os.path.join(BASE_PATH, "hypergraph_noisy.csv")

    region_info1.to_csv(csv_path1, index=False)
    print(f"\nSaved: {csv_path1}")
    print(f"  Columns: {list(region_info1.columns)}")
    print(f"  Rows: {len(region_info1)}")

    region_info2.to_csv(csv_path2, index=False)
    print(f"\nSaved: {csv_path2}")
    print(f"  Columns: {list(region_info2.columns)}")
    print(f"  Rows: {len(region_info2)}")

    # Verify data
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    print("\nClean hypergraph sample:")
    print(region_info1.head(3).to_string())

    print("\nNoisy hypergraph sample:")
    print(region_info2.head(3).to_string())

    # Check statistics
    print("\nClean statistics:")
    print(f"  Total regions: {len(region_info1)}")
    print(f"  Avg saddles per region: {region_info1['num_saddles'].mean():.2f}")
    print(f"  Avg hyperedge size: {region_info1['hyperedge_size'].mean():.2f}")

    print("\nNoisy statistics:")
    print(f"  Total regions: {len(region_info2)}")
    print(f"  Avg saddles per region: {region_info2['num_saddles'].mean():.2f}")
    print(f"  Avg hyperedge size: {region_info2['hyperedge_size'].mean():.2f}")

    # Generate visualization
    print("\n" + "=" * 60)
    print("GENERATING COOT INPUT VISUALIZATION")
    print("=" * 60)

    plot_coot_input(cp1, cp2, H1, H2, region_info1, region_info2,
                   os.path.join(BASE_PATH, "coot_input.png"))

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
