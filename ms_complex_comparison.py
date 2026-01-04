"""
Morse-Smale Complex Comparison using Optimal Transport Methods
VIS 2026 Research Project

Compares MS complexes from two sinusoidal surfaces (clean vs noisy) using:
1. Wasserstein Distance (WD) - feature-based matching
2. Gromov-Wasserstein Distance (GWD) - structure-preserving matching
3. Hypergraph COOT - combined feature and structural matching

Author: Research Team
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
import ot
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING AND PARSING
# =============================================================================

def load_critical_points(filepath: str) -> pd.DataFrame:
    """
    Load critical points from TTK-generated CSV.

    Returns DataFrame with columns:
    - point_id: Original point ID
    - cp_type: 0=minimum, 1=saddle, 2=maximum
    - x, y, z: Coordinates
    - scalar: Scalar field value
    """
    df = pd.read_csv(filepath)

    # Extract relevant columns
    result = pd.DataFrame({
        'point_id': df['Point ID'],
        'cp_type': df['CellDimension'],
        'x': df['Points_0'],
        'y': df['Points_1'],
        'z': df['Points_2'],
        'scalar': df['data']  # actual scalar field value
    })

    return result


def load_separatrices(filepath: str) -> pd.DataFrame:
    """
    Load separatrix data from TTK-generated CSV.

    The separatrices file contains all points along separatrix lines.
    CellDimension indicates whether it's a CP (0,2=extremum, 1=saddle) or line point.
    """
    df = pd.read_csv(filepath)
    return df


def load_segmentation(filepath: str) -> pd.DataFrame:
    """
    Load segmentation (ascending/descending manifolds) from TTK-generated CSV.

    Returns DataFrame with manifold assignments for each mesh vertex.
    """
    df = pd.read_csv(filepath)
    return df


def build_cp_graph_from_separatrices(cp_df: pd.DataFrame, sep_df: pd.DataFrame) -> np.ndarray:
    """
    Build adjacency matrix for critical points from separatrix data.

    In a Morse-Smale complex:
    - Saddles (type 1) connect to minima (type 0) and maxima (type 2)
    - We extract these connections from the separatrix trajectories

    Returns:
        Adjacency matrix (n_cp x n_cp) with 1s for connected CPs
    """
    n_cp = len(cp_df)
    adjacency = np.zeros((n_cp, n_cp))

    # Get CP indices by type
    cp_by_type = {
        0: cp_df[cp_df['cp_type'] == 0].index.tolist(),  # minima
        1: cp_df[cp_df['cp_type'] == 1].index.tolist(),  # saddles
        2: cp_df[cp_df['cp_type'] == 2].index.tolist()   # maxima
    }

    # In TTK separatrices, the separatrix lines connect CPs
    # Each separatrix has points with CellDimension indicating the type
    # CellDimension 1 typically marks saddle points, 0/2 mark extrema

    # Group consecutive points into separatrix segments
    # A segment starts at one CP type and ends at another

    # Extract unique separatrix endpoints by looking at CellDimension changes
    # For TTK data: we look for transitions between CP types

    sep_types = sep_df['CellDimension'].values
    sep_coords = sep_df[['Points_0', 'Points_1', 'Points_2']].values

    # Match separatrix endpoints to critical points
    cp_coords = cp_df[['x', 'y', 'z']].values

    # Find segments: sequences where type is constant or changes
    current_segment_start = 0

    for i in range(1, len(sep_types)):
        # Check if this is a segment boundary (type changes or gap)
        if sep_types[i] != sep_types[i-1] or i == len(sep_types) - 1:
            # End of segment - check if it's a CP (type 0, 1, or 2 in MS context)
            start_coord = sep_coords[current_segment_start]
            end_coord = sep_coords[i-1] if i > current_segment_start else sep_coords[i]

            # Find closest CPs to start and end
            start_dists = np.linalg.norm(cp_coords - start_coord, axis=1)
            end_dists = np.linalg.norm(cp_coords - end_coord, axis=1)

            start_cp = np.argmin(start_dists)
            end_cp = np.argmin(end_dists)

            # Only connect if distance is small (it's actually a CP)
            if start_dists[start_cp] < 1.0 and end_dists[end_cp] < 1.0:
                if start_cp != end_cp:
                    adjacency[start_cp, end_cp] = 1
                    adjacency[end_cp, start_cp] = 1

            current_segment_start = i

    # Alternative: use the saddle connectivity pattern
    # In a 2D MS complex, each saddle connects to exactly 2 minima and 2 maxima
    # Build this based on spatial proximity if graph is too sparse

    saddles = cp_df[cp_df['cp_type'] == 1]
    minima = cp_df[cp_df['cp_type'] == 0]
    maxima = cp_df[cp_df['cp_type'] == 2]

    if np.sum(adjacency) < len(saddles):  # Sparse graph, use proximity-based method
        saddle_coords = saddles[['x', 'y', 'z']].values
        min_coords = minima[['x', 'y', 'z']].values
        max_coords = maxima[['x', 'y', 'z']].values

        for s_idx, s_row in saddles.iterrows():
            s_coord = np.array([s_row['x'], s_row['y'], s_row['z']])

            # Find 2 closest minima
            if len(min_coords) > 0:
                min_dists = np.linalg.norm(min_coords - s_coord, axis=1)
                closest_mins = np.argsort(min_dists)[:2]
                for m_idx in closest_mins:
                    m_row_idx = minima.index[m_idx]
                    adjacency[s_idx, m_row_idx] = 1
                    adjacency[m_row_idx, s_idx] = 1

            # Find 2 closest maxima
            if len(max_coords) > 0:
                max_dists = np.linalg.norm(max_coords - s_coord, axis=1)
                closest_maxs = np.argsort(max_dists)[:2]
                for m_idx in closest_maxs:
                    m_row_idx = maxima.index[m_idx]
                    adjacency[s_idx, m_row_idx] = 1
                    adjacency[m_row_idx, s_idx] = 1

    return adjacency


def build_hypergraph_incidence(cp_df: pd.DataFrame, seg_df: pd.DataFrame,
                                sep_cells_df: pd.DataFrame = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build hypergraph incidence matrix from segmentation data.

    In a Morse-Smale complex, each 2-cell (region) is defined by EXACTLY:
    - 1 minimum (DescendingManifold = Point ID of the minimum)
    - 1 maximum (AscendingManifold + offset = Point ID of the maximum)
    - Saddles on the boundary that connect to BOTH this min AND this max via separatrices

    A REGION is a unique MorseSmaleManifold ID.

    Args:
        cp_df: Critical points DataFrame with 'point_id', 'cp_type'
        seg_df: Segmentation DataFrame with 'DescendingManifold', 'AscendingManifold', 'MorseSmaleManifold'
        sep_cells_df: Optional separatrices cells DataFrame for precise saddle connectivity.
                      If not provided, uses proximity-based saddle detection.

    Returns:
        H: Incidence matrix (n_cp x n_regions) where H[i,j] = 1 if CP i
           is on boundary of region j
        region_info: DataFrame with region metadata (min_id, max_id, size)
    """
    n_cp = len(cp_df)

    # Get CP info
    cp_types = cp_df['cp_type'].values
    cp_point_ids = cp_df['point_id'].values

    # Get CP indices by type
    minima_indices = np.where(cp_types == 0)[0]
    saddle_indices = np.where(cp_types == 1)[0]
    maxima_indices = np.where(cp_types == 2)[0]

    # Get CP Point IDs by type
    min_point_ids = set(cp_point_ids[minima_indices])
    saddle_point_ids = set(cp_point_ids[saddle_indices])
    max_point_ids = set(cp_point_ids[maxima_indices])

    # Find the offset for maxima (min of max Point IDs)
    max_offset = min(max_point_ids) if max_point_ids else 0

    # Build Point ID -> CP index mapping
    point_id_to_cp_idx = {pid: idx for idx, pid in enumerate(cp_point_ids)}

    # Filter out -1 (unassigned vertices) from segmentation
    valid_seg = seg_df[(seg_df['DescendingManifold'] >= 0) &
                       (seg_df['AscendingManifold'] >= 0)].copy()

    # Get unique regions from MorseSmaleManifold
    region_info = valid_seg.groupby('MorseSmaleManifold').agg({
        'DescendingManifold': 'first',
        'AscendingManifold': 'first',
        'Point ID': 'count'
    }).reset_index()
    region_info.columns = ['region_id', 'desc_manifold', 'asc_manifold', 'size']

    # Map manifold IDs to CP Point IDs
    # DescendingManifold value = Point ID of the minimum directly
    # AscendingManifold value + max_offset = Point ID of the maximum
    region_info['min_cp_id'] = region_info['desc_manifold']
    region_info['max_cp_id'] = region_info['asc_manifold'] + max_offset

    n_regions = len(region_info)

    # Build saddle connectivity from separatrices_cells if available
    saddle_to_mins = {}
    saddle_to_maxs = {}

    if sep_cells_df is not None:
        # Build CellId -> Point ID mapping from critical points
        cellid_to_cpid = dict(zip(cp_df['point_id'].map(lambda x: cp_df[cp_df['point_id'] == x]['point_id'].iloc[0]),
                                   cp_df['point_id']))
        # More robust: check if CellId column exists in cp_df original data
        # For now, use the point_id as a proxy

        # Get unique separatrices
        unique_sep = sep_cells_df.drop_duplicates(subset=['SeparatrixId'])[
            ['SeparatrixId', 'SourceId', 'DestinationId', 'SeparatrixType']
        ].copy()

        # SeparatrixType: 0 = descending (saddle -> min), 1 = ascending (saddle -> max)
        desc_sep = unique_sep[unique_sep['SeparatrixType'] == 0]
        asc_sep = unique_sep[unique_sep['SeparatrixType'] == 1]

        # Group by source (saddle) to find connected mins/maxs
        # Note: SourceId and DestinationId are CellIds, need proper mapping
        # For simplicity, assume direct correspondence for now
        saddle_to_mins = desc_sep.groupby('SourceId')['DestinationId'].apply(set).to_dict()
        saddle_to_maxs = asc_sep.groupby('SourceId')['DestinationId'].apply(set).to_dict()

    # Initialize incidence matrix
    H = np.zeros((n_cp, n_regions))

    for region_idx, row in region_info.iterrows():
        min_cp_id = int(row['min_cp_id'])
        max_cp_id = int(row['max_cp_id'])

        # Add minimum to hyperedge
        if min_cp_id in point_id_to_cp_idx:
            min_cp_idx = point_id_to_cp_idx[min_cp_id]
            H[min_cp_idx, region_idx] = 1

        # Add maximum to hyperedge
        if max_cp_id in point_id_to_cp_idx:
            max_cp_idx = point_id_to_cp_idx[max_cp_id]
            H[max_cp_idx, region_idx] = 1

        # Find boundary saddles
        if sep_cells_df is not None and len(saddle_to_mins) > 0:
            # Use separatrix connectivity: saddle must connect to both this min AND this max
            for saddle_id in saddle_point_ids:
                mins_connected = saddle_to_mins.get(saddle_id, set())
                maxs_connected = saddle_to_maxs.get(saddle_id, set())
                if min_cp_id in mins_connected and max_cp_id in maxs_connected:
                    if saddle_id in point_id_to_cp_idx:
                        H[point_id_to_cp_idx[saddle_id], region_idx] = 1
        else:
            # Fallback: use proximity-based saddle detection
            # This is less accurate but works without separatrices_cells data
            cp_coords = cp_df[['x', 'y', 'z']].values
            region_mask = valid_seg['MorseSmaleManifold'] == row['region_id']
            region_vertices = valid_seg[region_mask][['Points_0', 'Points_1', 'Points_2']].values

            if len(region_vertices) > 0:
                for s_idx in saddle_indices:
                    s_coord = cp_coords[s_idx]
                    dists = np.linalg.norm(region_vertices - s_coord, axis=1)
                    if np.min(dists) < 3.0:  # Proximity threshold
                        H[s_idx, region_idx] = 1

    # Update region_info with proper columns
    region_info_out = region_info[['region_id', 'min_cp_id', 'max_cp_id', 'size']].copy()
    region_info_out.columns = ['region_id', 'min_id', 'max_id', 'size']

    return H, region_info_out


def build_hypergraph_incidence_simple(cp_df: pd.DataFrame, seg_df: pd.DataFrame) -> np.ndarray:
    """
    Wrapper that returns only the incidence matrix for backward compatibility.
    """
    H, _ = build_hypergraph_incidence(cp_df, seg_df)
    return H


# =============================================================================
# METHOD 1: WASSERSTEIN DISTANCE
# =============================================================================

def compute_wasserstein_distance(
    cp1: pd.DataFrame,
    cp2: pd.DataFrame
) -> Tuple[float, np.ndarray]:
    """
    Compute Wasserstein distance between two MS complexes based on SCALAR VALUES ONLY.

    This is the simplest OT formulation:
    - Cost C[i,j] = |f_i - f_j| (absolute difference of scalar values)
    - Ignores structure completely
    - Only cares about matching CPs with similar function values

    Args:
        cp1, cp2: Critical point DataFrames

    Returns:
        distance: Wasserstein distance
        coupling: Optimal transport coupling matrix
    """
    n1, n2 = len(cp1), len(cp2)

    # Extract scalar values
    scalar1 = cp1['scalar'].values
    scalar2 = cp2['scalar'].values

    # Cost matrix: absolute difference of scalar values |f_i - f_j|
    C = np.abs(scalar1[:, None] - scalar2[None, :])

    # Normalize cost matrix to [0, 1]
    C = C / (C.max() + 1e-8)

    # Uniform distributions
    a = np.ones(n1) / n1
    b = np.ones(n2) / n2

    # Compute optimal transport
    coupling = ot.emd(a, b, C)
    distance = ot.emd2(a, b, C)

    return distance, coupling


# =============================================================================
# METHOD 2: GROMOV-WASSERSTEIN DISTANCE
# =============================================================================

def compute_gromov_wasserstein_distance(
    cp1: pd.DataFrame,
    cp2: pd.DataFrame,
    adj1: np.ndarray,
    adj2: np.ndarray,
    use_shortest_path: bool = True
) -> Tuple[float, np.ndarray]:
    """
    Compute Gromov-Wasserstein distance between two MS complexes.

    Uses graph structure (adjacency) to compare internal distances.

    Args:
        cp1, cp2: Critical point DataFrames
        adj1, adj2: Adjacency matrices
        use_shortest_path: If True, use shortest path distances; else use adjacency

    Returns:
        distance: GW distance
        coupling: Optimal transport coupling matrix
    """
    n1, n2 = len(cp1), len(cp2)

    # Compute internal distance matrices
    if use_shortest_path:
        # Convert adjacency to shortest path distances
        # Replace 0s with inf for non-edges (except diagonal)
        adj1_sp = adj1.copy()
        adj1_sp[adj1_sp == 0] = np.inf
        np.fill_diagonal(adj1_sp, 0)

        adj2_sp = adj2.copy()
        adj2_sp[adj2_sp == 0] = np.inf
        np.fill_diagonal(adj2_sp, 0)

        D1 = shortest_path(adj1_sp, directed=False)
        D2 = shortest_path(adj2_sp, directed=False)

        # Handle disconnected components (inf values)
        max_dist1 = D1[np.isfinite(D1)].max() if np.any(np.isfinite(D1)) else 1
        max_dist2 = D2[np.isfinite(D2)].max() if np.any(np.isfinite(D2)) else 1
        D1[np.isinf(D1)] = max_dist1 + 1
        D2[np.isinf(D2)] = max_dist2 + 1
    else:
        D1 = adj1.copy()
        D2 = adj2.copy()

    # Normalize distance matrices
    D1 = D1 / (D1.max() + 1e-8)
    D2 = D2 / (D2.max() + 1e-8)

    # Uniform distributions
    p = np.ones(n1) / n1
    q = np.ones(n2) / n2

    # Compute Gromov-Wasserstein
    coupling, log = ot.gromov.gromov_wasserstein(
        D1, D2, p, q,
        loss_fun='square_loss',
        log=True
    )

    distance = log['gw_dist']

    return distance, coupling


# =============================================================================
# METHOD 3: HYPERGRAPH COOT (CO-OPTIMAL TRANSPORT)
# =============================================================================

def compute_hypergraph_coot(
    cp1: pd.DataFrame,
    cp2: pd.DataFrame,
    H1: np.ndarray,
    H2: np.ndarray,
    alpha: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Hypergraph COOT distance between two MS complexes.

    COOT (Co-Optimal Transport) jointly optimizes:
    1. Transport between CPs (sample coupling)
    2. Transport between regions/hyperedges (feature coupling)

    The cost combines:
    - Feature cost: how similar are CP features
    - Structural cost: how well does the hypergraph structure match

    Args:
        cp1, cp2: Critical point DataFrames
        H1, H2: Hypergraph incidence matrices (n_cp x n_regions)
        alpha: Weight for feature vs structure (0=pure structure, 1=pure feature)
        max_iter: Maximum iterations for alternating optimization
        tol: Convergence tolerance

    Returns:
        distance: COOT distance
        sample_coupling: Transport plan between CPs
        feature_coupling: Transport plan between regions
    """
    n1, n2 = len(cp1), len(cp2)
    m1, m2 = H1.shape[1], H2.shape[1]  # Number of regions

    # Build feature cost matrix for CPs
    scalar1 = cp1['scalar'].values.reshape(-1, 1)
    scalar2 = cp2['scalar'].values.reshape(-1, 1)

    # Normalize
    scalar1 = (scalar1 - scalar1.mean()) / (scalar1.std() + 1e-8)
    scalar2 = (scalar2 - scalar2.mean()) / (scalar2.std() + 1e-8)

    C_feature = cdist(scalar1, scalar2, metric='euclidean')
    C_feature = C_feature / (C_feature.max() + 1e-8)

    # Distributions
    p_sample = np.ones(n1) / n1  # CP distribution for complex 1
    q_sample = np.ones(n2) / n2  # CP distribution for complex 2
    p_feature = np.ones(m1) / m1  # Region distribution for complex 1
    q_feature = np.ones(m2) / m2  # Region distribution for complex 2

    # Initialize couplings
    pi_sample = np.outer(p_sample, q_sample)  # n1 x n2
    pi_feature = np.outer(p_feature, q_feature)  # m1 x m2

    prev_cost = np.inf

    for iteration in range(max_iter):
        # Step 1: Update sample coupling (CPs) given feature coupling (regions)
        # Structure cost based on hypergraph: how well do region memberships align
        C_structure = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                # Cost based on how different the hypergraph neighborhoods are
                # Using the current feature coupling to align regions
                h1_i = H1[i, :]  # Region memberships for CP i in complex 1
                h2_j = H2[j, :]  # Region memberships for CP j in complex 2

                # Transport the region memberships and compare
                h1_transported = h1_i @ pi_feature  # Transport h1 to space of complex 2
                h2_normalized = h2_j / (h2_j.sum() + 1e-8)
                h1_transported_normalized = h1_transported / (h1_transported.sum() + 1e-8)

                # L2 distance between transported memberships
                C_structure[i, j] = np.linalg.norm(h1_transported_normalized - h2_normalized)

        C_structure = C_structure / (C_structure.max() + 1e-8)

        # Combined cost
        C_sample = alpha * C_feature + (1 - alpha) * C_structure

        # Solve OT for sample coupling
        pi_sample = ot.emd(p_sample, q_sample, C_sample)

        # Step 2: Update feature coupling (regions) given sample coupling (CPs)
        # Cost based on which CPs belong to which regions
        C_region = np.zeros((m1, m2))

        for r1 in range(m1):
            for r2 in range(m2):
                # Which CPs belong to region r1 and r2
                cps_in_r1 = H1[:, r1]  # n1 vector
                cps_in_r2 = H2[:, r2]  # n2 vector

                # Transport CPs from complex 1 to complex 2 and compare
                cps_transported = cps_in_r1 @ pi_sample  # Transport to complex 2 space
                cps_transported_normalized = cps_transported / (cps_transported.sum() + 1e-8)
                cps_in_r2_normalized = cps_in_r2 / (cps_in_r2.sum() + 1e-8)

                C_region[r1, r2] = np.linalg.norm(cps_transported_normalized - cps_in_r2_normalized)

        C_region = C_region / (C_region.max() + 1e-8)

        # Solve OT for feature coupling
        pi_feature = ot.emd(p_feature, q_feature, C_region)

        # Compute total cost
        cost_feature = np.sum(C_feature * pi_sample)
        cost_structure = np.sum(C_structure * pi_sample)
        total_cost = alpha * cost_feature + (1 - alpha) * cost_structure

        # Check convergence
        if abs(prev_cost - total_cost) < tol:
            break
        prev_cost = total_cost

    return total_cost, pi_sample, pi_feature


# =============================================================================
# MAIN COMPARISON PIPELINE
# =============================================================================

class MSComplexComparator:
    """
    Main class for comparing Morse-Smale complexes using optimal transport.
    """

    def __init__(self,
                 cp_file1: str, sep_file1: str, seg_file1: str,
                 cp_file2: str, sep_file2: str, seg_file2: str,
                 sep_cells_file1: str = None, sep_cells_file2: str = None):
        """
        Initialize comparator with file paths for two MS complexes.

        Args:
            sep_cells_file1, sep_cells_file2: Optional paths to separatrices_cells.csv files
                which contain precise connectivity info (SourceId, DestinationId, SeparatrixType).
                If not provided, proximity-based saddle detection is used.
        """
        print("Loading MS Complex 1...")
        self.cp1 = load_critical_points(cp_file1)
        self.sep1 = load_separatrices(sep_file1)
        self.seg1 = load_segmentation(seg_file1)

        # Load separatrices cells if available
        self.sep_cells1 = None
        if sep_cells_file1:
            import os
            if os.path.exists(sep_cells_file1):
                self.sep_cells1 = pd.read_csv(sep_cells_file1)
                print(f"  Loaded separatrices_cells: {len(self.sep_cells1)} rows")

        print("Loading MS Complex 2...")
        self.cp2 = load_critical_points(cp_file2)
        self.sep2 = load_separatrices(sep_file2)
        self.seg2 = load_segmentation(seg_file2)

        # Load separatrices cells if available
        self.sep_cells2 = None
        if sep_cells_file2:
            import os
            if os.path.exists(sep_cells_file2):
                self.sep_cells2 = pd.read_csv(sep_cells_file2)
                print(f"  Loaded separatrices_cells: {len(self.sep_cells2)} rows")

        print(f"Complex 1: {len(self.cp1)} CPs ({sum(self.cp1['cp_type']==0)} min, "
              f"{sum(self.cp1['cp_type']==1)} saddle, {sum(self.cp1['cp_type']==2)} max)")
        print(f"Complex 2: {len(self.cp2)} CPs ({sum(self.cp2['cp_type']==0)} min, "
              f"{sum(self.cp2['cp_type']==1)} saddle, {sum(self.cp2['cp_type']==2)} max)")

        # Build graph structures
        print("\nBuilding graph adjacency from separatrices...")
        self.adj1 = build_cp_graph_from_separatrices(self.cp1, self.sep1)
        self.adj2 = build_cp_graph_from_separatrices(self.cp2, self.sep2)
        print(f"Complex 1 edges: {int(np.sum(self.adj1) / 2)}")
        print(f"Complex 2 edges: {int(np.sum(self.adj2) / 2)}")

        # Build hypergraph structures
        print("\nBuilding hypergraph incidence from segmentation...")
        self.H1, self.region_info1 = build_hypergraph_incidence(self.cp1, self.seg1, self.sep_cells1)
        self.H2, self.region_info2 = build_hypergraph_incidence(self.cp2, self.seg2, self.sep_cells2)
        print(f"Complex 1 regions: {self.H1.shape[1]}")
        print(f"Complex 2 regions: {self.H2.shape[1]}")

    def compare_wasserstein(self) -> Dict:
        """
        Compare using Wasserstein distance (scalar values only).

        Cost: C[i,j] = |f_i - f_j|
        - Ignores structure completely
        - Only matches by scalar value similarity
        """
        print("\n" + "="*60)
        print("METHOD 1: WASSERSTEIN DISTANCE (Scalar Only)")
        print("="*60)
        print("Cost: C[i,j] = |f_i - f_j| (absolute scalar difference)")

        distance, coupling = compute_wasserstein_distance(self.cp1, self.cp2)

        print(f"\nWasserstein Distance: {distance:.6f}")

        # Extract correspondences
        correspondences = self._extract_correspondences(coupling)
        print(f"Number of significant correspondences: {len(correspondences)}")

        return {
            'method': 'Wasserstein',
            'distance': distance,
            'coupling': coupling,
            'correspondences': correspondences
        }

    def compare_gromov_wasserstein(self,
                                   use_shortest_path: bool = True
                                   ) -> Dict:
        """
        Compare using Gromov-Wasserstein distance.
        """
        print("\n" + "="*60)
        print("METHOD 2: GROMOV-WASSERSTEIN DISTANCE")
        print("="*60)

        print(f"Using shortest path distances: {use_shortest_path}")

        distance, coupling = compute_gromov_wasserstein_distance(
            self.cp1, self.cp2, self.adj1, self.adj2, use_shortest_path
        )

        print(f"\nGromov-Wasserstein Distance: {distance:.6f}")

        # Extract correspondences
        correspondences = self._extract_correspondences(coupling)
        print(f"Number of significant correspondences: {len(correspondences)}")

        return {
            'method': 'Gromov-Wasserstein',
            'distance': distance,
            'coupling': coupling,
            'correspondences': correspondences
        }

    def compare_hypergraph_coot(self,
                                alpha: float = 0.5,
                                max_iter: int = 100
                                ) -> Dict:
        """
        Compare using Hypergraph COOT.
        """
        print("\n" + "="*60)
        print("METHOD 3: HYPERGRAPH COOT")
        print("="*60)

        print(f"Alpha (feature vs structure): {alpha}")
        print(f"Max iterations: {max_iter}")

        distance, sample_coupling, feature_coupling = compute_hypergraph_coot(
            self.cp1, self.cp2, self.H1, self.H2, alpha, max_iter
        )

        print(f"\nCOOT Distance: {distance:.6f}")

        # Extract correspondences
        cp_correspondences = self._extract_correspondences(sample_coupling)
        region_correspondences = self._extract_correspondences(feature_coupling)

        print(f"CP correspondences: {len(cp_correspondences)}")
        print(f"Region correspondences: {len(region_correspondences)}")

        return {
            'method': 'Hypergraph-COOT',
            'distance': distance,
            'sample_coupling': sample_coupling,
            'feature_coupling': feature_coupling,
            'cp_correspondences': cp_correspondences,
            'region_correspondences': region_correspondences
        }

    def _extract_correspondences(self, coupling: np.ndarray,
                                  threshold: float = None) -> List[Tuple[int, int, float]]:
        """
        Extract significant correspondences from coupling matrix.
        Uses adaptive threshold based on uniform coupling.
        """
        # Adaptive threshold: values significantly above uniform coupling
        n1, n2 = coupling.shape
        uniform_val = 1.0 / (n1 * n2)
        if threshold is None:
            threshold = uniform_val * 0.5  # Half of uniform as minimum threshold

        correspondences = []
        for i in range(coupling.shape[0]):
            for j in range(coupling.shape[1]):
                if coupling[i, j] > threshold:
                    correspondences.append((i, j, coupling[i, j]))

        # Sort by coupling strength
        correspondences.sort(key=lambda x: -x[2])
        return correspondences

    def run_all_comparisons(self) -> Dict:
        """
        Run all three comparison methods.

        Clean separation:
        - WD: Only scalar values (ignores structure)
        - GWD: Only graph structure (ignores scalar values)
        - COOT: Hypergraph structure + scalar features
        """
        results = {}

        # Method 1: Wasserstein (scalar only)
        results['WD'] = self.compare_wasserstein()

        # Method 2: Gromov-Wasserstein (graph structure only)
        results['GWD'] = self.compare_gromov_wasserstein(use_shortest_path=True)

        # Method 3: COOT with different alpha values
        # alpha=1.0 would be pure feature, alpha=0.0 would be pure structure
        results['COOT_0.5'] = self.compare_hypergraph_coot(alpha=0.5)

        return results

    def print_summary(self, results: Dict):
        """
        Print summary of all comparison results.
        """
        print("\n" + "="*60)
        print("SUMMARY OF ALL COMPARISONS")
        print("="*60)
        print(f"\n{'Method':<25} {'Distance':<15} {'Correspondences':<15}")
        print("-"*55)

        for name, res in results.items():
            n_corr = len(res.get('correspondences', res.get('cp_correspondences', [])))
            print(f"{name:<25} {res['distance']:.6f}       {n_corr}")

        # Find best method (smallest distance suggests most similar)
        print("\n" + "-"*55)
        min_dist = min(res['distance'] for res in results.values())
        max_dist = max(res['distance'] for res in results.values())
        print(f"Distance range: [{min_dist:.6f}, {max_dist:.6f}]")

    def print_top_correspondences(self, results: Dict, method_key: str, n_top: int = 10):
        """
        Print top correspondences with CP details.
        """
        print(f"\n--- Top {n_top} correspondences for {method_key} ---")

        res = results[method_key]
        corrs = res.get('correspondences', res.get('cp_correspondences', []))[:n_top]

        cp_types = {0: 'min', 1: 'sad', 2: 'max'}

        print(f"{'CP1 idx':<10} {'Type1':<6} {'Scalar1':<10} {'CP2 idx':<10} {'Type2':<6} {'Scalar2':<10} {'Weight':<10}")
        print("-"*72)

        for cp1_idx, cp2_idx, weight in corrs:
            cp1 = self.cp1.iloc[cp1_idx]
            cp2 = self.cp2.iloc[cp2_idx]
            print(f"{cp1_idx:<10} {cp_types[cp1['cp_type']]:<6} {cp1['scalar']:<10.4f} "
                  f"{cp2_idx:<10} {cp_types[cp2['cp_type']]:<6} {cp2['scalar']:<10.4f} {weight:<10.6f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import os

    # File paths
    base_path = "/Users/gmeng/Desktop/COOT on Morse-Smale"

    # Clean surface files
    cp_clean = os.path.join(base_path, "sinusoidal0_critical_points.csv")
    sep_clean = os.path.join(base_path, "sinusoidal0_separatrices.csv")
    seg_clean = os.path.join(base_path, "sinusoidal0_segmentation.csv")
    sep_cells_clean = os.path.join(base_path, "separatrices_cells.csv")  # Contains connectivity info

    # Noisy surface files
    cp_noise = os.path.join(base_path, "sinusoidal0_critical_points_noise.csv")
    sep_noise = os.path.join(base_path, "sinusoidal0_separatrices_noise.csv")
    seg_noise = os.path.join(base_path, "sinusoidal0_segmentation_noise.csv")
    sep_cells_noise = None  # Not available - will use proximity-based saddle detection

    # Initialize comparator
    comparator = MSComplexComparator(
        cp_clean, sep_clean, seg_clean,
        cp_noise, sep_noise, seg_noise,
        sep_cells_file1=sep_cells_clean,
        sep_cells_file2=sep_cells_noise
    )

    # Run all comparisons
    results = comparator.run_all_comparisons()

    # Print summary
    comparator.print_summary(results)

    # Print top correspondences for each method
    comparator.print_top_correspondences(results, 'WD', n_top=10)
    comparator.print_top_correspondences(results, 'GWD', n_top=10)
    comparator.print_top_correspondences(results, 'COOT_0.5', n_top=10)

    # Save correspondences to files for visualization
    print("\n" + "="*60)
    print("SAVING CORRESPONDENCES")
    print("="*60)

    # Save WD correspondences
    wd_corr = results['WD']['correspondences'][:20]  # Top 20
    with open(os.path.join(base_path, "correspondences_wd.csv"), 'w') as f:
        f.write("cp1_idx,cp2_idx,weight\n")
        for c in wd_corr:
            f.write(f"{c[0]},{c[1]},{c[2]:.6f}\n")
    print("Saved: correspondences_wd.csv")

    # Save GWD correspondences
    gwd_corr = results['GWD']['correspondences'][:20]
    with open(os.path.join(base_path, "correspondences_gwd.csv"), 'w') as f:
        f.write("cp1_idx,cp2_idx,weight\n")
        for c in gwd_corr:
            f.write(f"{c[0]},{c[1]},{c[2]:.6f}\n")
    print("Saved: correspondences_gwd.csv")

    # Save COOT correspondences
    coot_corr = results['COOT_0.5']['cp_correspondences'][:20]
    with open(os.path.join(base_path, "correspondences_coot.csv"), 'w') as f:
        f.write("cp1_idx,cp2_idx,weight\n")
        for c in coot_corr:
            f.write(f"{c[0]},{c[1]},{c[2]:.6f}\n")
    print("Saved: correspondences_coot.csv")

    print("\nDone!")
