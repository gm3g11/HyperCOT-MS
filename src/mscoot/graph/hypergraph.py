"""Hypergraph construction from Morse-Smale complex."""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Optional, Tuple


def build_saddle_connectivity(sep_df: pd.DataFrame, saddle_cell_ids: set,
                              onesaddle_cell_ids: set = None,
                              src_col: str = 'SourceId',
                              dst_col: str = 'DestinationId',
                              type_col: str = 'SeparatrixType',
                              dedup_cols: Optional[list] = None):
    """Build saddle connectivity from separatrices.

    For 2D: saddle_cell_ids are 1-saddles (dim=1). Direct connectivity.
    For 3D: saddle_cell_ids are 2-saddles (dim=2), onesaddle_cell_ids are
        1-saddles (dim=1). 2-saddle→min is chained through 1-saddles.

    Returns:
        saddle_to_mins: saddle CellId -> set of connected min CellIds
        saddle_to_maxs: saddle CellId -> set of connected max CellIds
    """
    saddle_to_mins = defaultdict(set)
    saddle_to_maxs = defaultdict(set)

    if dedup_cols is None:
        dedup_cols = [src_col, dst_col, type_col]
    available = [c for c in dedup_cols if c in sep_df.columns]
    unique_sep = sep_df.drop_duplicates(subset=available) if available else sep_df

    if onesaddle_cell_ids is not None:
        # 3D mode: chain 2-saddle → min through 1-saddles
        # Type 0: 1-saddle → min
        # Type 1: 1-saddle → 2-saddle
        # Type 2: 2-saddle → max
        onesad_to_mins = defaultdict(set)
        twosad_to_onesads = defaultdict(set)

        for _, row in unique_sep.iterrows():
            src_cell = int(row[src_col])
            dst_cell = int(row[dst_col])
            sep_type = int(row[type_col])

            if sep_type == 0 and src_cell in onesaddle_cell_ids:
                onesad_to_mins[src_cell].add(dst_cell)
            elif sep_type == 1 and src_cell in onesaddle_cell_ids and dst_cell in saddle_cell_ids:
                twosad_to_onesads[dst_cell].add(src_cell)
            elif sep_type == 2 and src_cell in saddle_cell_ids:
                saddle_to_maxs[src_cell].add(dst_cell)

        # Chain: 2-saddle → 1-saddle(s) → min(s)
        for twosad, onesads in twosad_to_onesads.items():
            for onesad in onesads:
                saddle_to_mins[twosad].update(onesad_to_mins.get(onesad, set()))
    else:
        # 2D mode: direct connectivity
        for _, row in unique_sep.iterrows():
            src_cell = int(row[src_col])
            dst_cell = int(row[dst_col])
            sep_type = int(row[type_col])

            if src_cell in saddle_cell_ids:
                if sep_type == 0:  # Descending: saddle -> min
                    saddle_to_mins[src_cell].add(dst_cell)
                else:  # Ascending: saddle -> max
                    saddle_to_maxs[src_cell].add(dst_cell)

    return saddle_to_mins, saddle_to_maxs


def _sort_saddles_angularly(saddle_indices: List[int], min_pos: np.ndarray,
                            max_pos: np.ndarray, saddle_positions: np.ndarray) -> List[int]:
    """Sort boundary saddles by angle around the region center.

    For 2D: uses atan2 directly.
    For 3D: projects to local 2D plane via PCA, then uses atan2.

    Args:
        saddle_indices: Saddle indices (into the saddles-only array).
        min_pos: Position of the region's minimum.
        max_pos: Position of the region's maximum.
        saddle_positions: (n_saddles, ndim) positions of ALL saddles.

    Returns:
        Saddle indices sorted by angle around the center.
    """
    center = (min_pos + max_pos) / 2.0
    ndim = saddle_positions.shape[1]

    if ndim <= 2:
        # 2D: direct atan2
        angles = []
        for s_idx in saddle_indices:
            s_pos = saddle_positions[s_idx]
            angle = np.arctan2(s_pos[1] - center[1], s_pos[0] - center[0])
            angles.append(angle)
    else:
        # 3D: project to local 2D frame via SVD
        pts = np.array([saddle_positions[s_idx] for s_idx in saddle_indices])
        centered = pts - center
        if len(centered) >= 2:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            # Project onto first two principal components
            proj = centered @ Vt[:2].T
            angles = [np.arctan2(p[1], p[0]) for p in proj]
        else:
            angles = [0.0] * len(saddle_indices)

    order = np.argsort(angles)
    return [saddle_indices[i] for i in order]


def _build_hypergraph_3d_twosad_regions(cp_df, sep_df, pers_df=None,
                                         min_persistence=0.0):
    """Build 3D hypergraph with 2-saddles as region centers.

    In 3D, TTK cannot simplify type-1 (1sad-2sad) pairs, leaving ~150
    unsimplifiable saddle pairs regardless of threshold.  With only 1 max,
    the standard approach (regions around maxima) yields a single trivial
    region.

    This function drops one Morse level:
        region center = 2-saddle  (dim=2)
        boundary CP    = 1-saddle (dim=1)
        anchor         = minimum  (dim=0)

    Each region is defined by a 2-saddle and the 1-saddles connected to it
    via type-1 separatrices.  The hyperedge contains: [min, 1-saddles..., 2-saddle].
    Global CP indexing follows TTK order (dim0, dim1, dim2, dim3).

    When pers_df is provided, only 2-saddles involved in type-1 pairs with
    persistence > min_persistence are kept as region centers.  83% of type-1
    pairs have zero persistence (grid artifacts); filtering to > 0 gives
    ~25 meaningful regions.

    Args:
        cp_df: Critical points DataFrame.
        sep_df: Separatrices DataFrame.
        pers_df: Persistence diagram DataFrame (optional).
        min_persistence: Minimum type-1 persistence for a 2-saddle to be
            a region center (default: 0.0 keeps all non-zero pairs).

    Returns:
        DataFrame with standard hypergraph columns, or None.
    """
    cp_df = cp_df.reset_index(drop=True)

    minima = cp_df[cp_df['CellDimension'] == 0].reset_index(drop=True)
    onesaddles = cp_df[cp_df['CellDimension'] == 1].reset_index(drop=True)
    twosaddles = cp_df[cp_df['CellDimension'] == 2].reset_index(drop=True)
    # dim=3 maxima are ignored in this scheme

    n_min = len(minima)
    n_1sad = len(onesaddles)
    n_2sad = len(twosaddles)

    if n_min == 0 or n_2sad == 0 or n_1sad == 0:
        return None

    # Global offsets (TTK orders: dim0, dim1, dim2, dim3)
    onesad_offset = n_min
    twosad_offset = n_min + n_1sad

    # Build CellId -> local index maps
    onesad_cell_to_idx = {}
    for idx, row in onesaddles.iterrows():
        onesad_cell_to_idx[int(row['CellId'])] = idx
    twosad_cell_to_idx = {}
    for idx, row in twosaddles.iterrows():
        twosad_cell_to_idx[int(row['CellId'])] = idx
    min_cell_to_idx = {}
    for idx, row in minima.iterrows():
        min_cell_to_idx[int(row['CellId'])] = idx

    onesad_cell_ids = set(onesad_cell_to_idx.keys())
    twosad_cell_ids = set(twosad_cell_to_idx.keys())

    # Parse separatrices to build connectivity:
    #   type 0: 1-saddle -> min
    #   type 1: 1-saddle -> 2-saddle  (defines region membership)
    #   type 2: 2-saddle -> max       (ignored in this scheme)
    src_col, dst_col, type_col = 'SourceId', 'DestinationId', 'SeparatrixType'
    dedup_cols = [src_col, dst_col, type_col]
    available = [c for c in dedup_cols if c in sep_df.columns]
    unique_sep = sep_df.drop_duplicates(subset=available) if available else sep_df

    # 2-saddle -> set of connected 1-saddle CellIds (from type-1 seps)
    twosad_to_onesads = defaultdict(set)
    # 1-saddle -> set of connected min CellIds (from type-0 seps)
    onesad_to_mins = defaultdict(set)

    for _, row in unique_sep.iterrows():
        src = int(row[src_col])
        dst = int(row[dst_col])
        stype = int(row[type_col])

        if stype == 0 and src in onesad_cell_ids:
            onesad_to_mins[src].add(dst)
        elif stype == 1 and src in onesad_cell_ids and dst in twosad_cell_ids:
            twosad_to_onesads[dst].add(src)

    # Filter 2-saddles by type-1 persistence if diagram provided.
    # TTK's PairIdentifier doesn't map directly to CellId, so we use the
    # count of significant pairs to select the top-connected 2-saddles.
    max_regions = len(twosad_to_onesads)  # default: keep all
    if pers_df is not None and len(pers_df) > 0:
        t1 = pers_df[pers_df['PairType'] == 1]
        if len(t1) > 0:
            n_significant = int((t1['Persistence'] > min_persistence).sum())
            # Add 1 for the unpaired 2-saddle (infinite persistence)
            max_regions = n_significant + 1
            max_regions = max(max_regions, 1)

    # Select top 2-saddles by connectivity (number of 1-saddle neighbors)
    ranked = sorted(twosad_to_onesads.items(),
                    key=lambda x: len(x[1]), reverse=True)
    selected = ranked[:max_regions]

    # Build one region per selected 2-saddle
    rows = []
    for region_id, (twosad_cell, onesad_cells) in enumerate(selected, start=1):

        twosad_idx = twosad_cell_to_idx[twosad_cell]

        # Boundary 1-saddles (local indices into onesaddles DataFrame)
        boundary = sorted(onesad_cell_to_idx[c] for c in onesad_cells
                          if c in onesad_cell_to_idx)
        if not boundary:
            continue

        # Find connected minima (chain: 2-saddle -> 1-saddles -> mins)
        connected_mins = set()
        for osad_cell in onesad_cells:
            connected_mins.update(onesad_to_mins.get(osad_cell, set()))
        # Pick the first valid minimum (usually there's only one global min)
        min_idx = None
        for mc in connected_mins:
            if mc in min_cell_to_idx:
                min_idx = min_cell_to_idx[mc]
                break
        if min_idx is None and len(min_cell_to_idx) > 0:
            min_idx = next(iter(min_cell_to_idx.values()))
        if min_idx is None:
            continue

        # Global indices: [min, 1-saddles..., 2-saddle]
        hyperedge = ([min_idx] +
                     [s + onesad_offset for s in boundary] +
                     [twosad_idx + twosad_offset])

        rows.append({
            'region_id': region_id,
            'min_id': min_idx,
            'max_id': twosad_idx,  # "max" is the 2-saddle in this scheme
            'num_saddles': len(boundary),
            'boundary_saddles': str(boundary),
            'hyperedge': str(hyperedge),
            'hyperedge_size': len(hyperedge),
            'region_size': 1,  # no segmentation-based size for this scheme
        })

    return pd.DataFrame(rows) if rows else None


def build_hypergraph(cp_df: pd.DataFrame, seg_df: pd.DataFrame,
                     sep_df: Optional[pd.DataFrame] = None,
                     split_large: bool = True,
                     pers_df: Optional[pd.DataFrame] = None,
                     min_persistence: float = 0.0) -> Optional[pd.DataFrame]:
    """Build hypergraph for a single timestep.

    Each hyperedge represents a Morse-Smale region: 1 min + boundary saddles + 1 max.
    Boundary saddles identified via separatrix connectivity.

    Supports both 2D (CellDimension 0/1/2) and 3D (CellDimension 0/1/2/3).
    In 3D, uses 2-saddles as region centers and 1-saddles as boundary CPs
    (dropped-level scheme), since TTK cannot simplify type-1 pairs.
    Splitting is disabled for 3D (ring topology doesn't hold).

    Args:
        cp_df: Critical points with columns [CellDimension, CellId, ...].
        seg_df: Segmentation with columns [MorseSmaleManifold, DescendingManifold, AscendingManifold].
        sep_df: Separatrices with columns [SourceId, DestinationId, SeparatrixType].
        split_large: If True (2D only), split regions with 3+ boundary saddles into
            standard 4-CP sub-regions (1 min + 2 saddles + 1 max each).
        pers_df: Persistence diagram for this timestep (3D only, optional).
        min_persistence: Minimum type-1 persistence for 3D region filtering (default: 0.0).

    Returns:
        DataFrame with columns: region_id, min_id, max_id, num_saddles,
            boundary_saddles, hyperedge, hyperedge_size, region_size.
        None if no valid hyperedges found.
    """
    cp_df = cp_df.reset_index(drop=True)

    # Auto-detect 3D: CellDimension 3 present means 4 CP types
    is_3d = (cp_df['CellDimension'] == 3).any()

    if is_3d:
        # 3D: use dropped-level scheme (2-saddles as region centers)
        return _build_hypergraph_3d_twosad_regions(cp_df, sep_df, pers_df,
                                                    min_persistence)

    # 2D: dim 0 = min, dim 1 = saddle, dim 2 = max
    minima = cp_df[cp_df['CellDimension'] == 0].reset_index(drop=True)
    onesaddles = None
    saddles = cp_df[cp_df['CellDimension'] == 1].reset_index(drop=True)
    maxima = cp_df[cp_df['CellDimension'] == 2].reset_index(drop=True)
    n_min = len(minima)
    n_1sad = 0
    n_sad = len(saddles)
    n_max = len(maxima)
    saddle_offset = n_min
    max_offset = n_min + n_sad

    if n_min == 0 or n_max == 0:
        return None

    # Get positions for angular sorting (2D or 3D)
    x_col = [c for c in cp_df.columns if c in ('Points:0', 'Points_0', 'x')][0]
    y_col = [c for c in cp_df.columns if c in ('Points:1', 'Points_1', 'y')][0]
    z_candidates = [c for c in cp_df.columns if c in ('Points:2', 'Points_2', 'z')]
    z_col = z_candidates[0] if z_candidates else None

    # Use 3D if z column exists and has non-zero values
    coord_cols = [x_col, y_col]
    if z_col is not None and (cp_df[z_col].abs() > 1e-10).any():
        coord_cols.append(z_col)

    min_positions = minima[coord_cols].values
    saddle_positions = saddles[coord_cols].values
    max_positions = maxima[coord_cols].values

    min_cell_to_idx = {int(row['CellId']): idx for idx, row in minima.iterrows()}
    max_cell_to_idx = {int(row['CellId']): idx for idx, row in maxima.iterrows()}
    saddle_cell_to_idx = {int(row['CellId']): idx for idx, row in saddles.iterrows()}

    min_idx_to_cell = {v: k for k, v in min_cell_to_idx.items()}
    max_idx_to_cell = {v: k for k, v in max_cell_to_idx.items()}

    saddle_cell_ids = set(saddle_cell_to_idx.keys())

    # Build saddle connectivity
    if sep_df is not None and len(sep_df) > 0:
        saddle_to_mins, saddle_to_maxs = build_saddle_connectivity(
            sep_df, saddle_cell_ids
        )
    else:
        saddle_to_mins = defaultdict(set)
        saddle_to_maxs = defaultdict(set)
        for sc in saddle_cell_ids:
            for mc in min_cell_to_idx:
                saddle_to_mins[sc].add(mc)
            for mc in max_cell_to_idx:
                saddle_to_maxs[sc].add(mc)

    # Get unique regions from segmentation
    valid_seg = seg_df[
        (seg_df['DescendingManifold'] >= 0) & (seg_df['AscendingManifold'] >= 0)
    ].copy()

    region_groups = valid_seg.groupby('MorseSmaleManifold').agg({
        'DescendingManifold': 'first',
        'AscendingManifold': 'first'
    }).reset_index()

    region_sizes = seg_df.groupby('MorseSmaleManifold').size().to_dict()

    rows = []
    sub_id_counter = 0
    for _, region in region_groups.iterrows():
        region_id = int(region['MorseSmaleManifold'])
        min_idx = int(region['DescendingManifold'])
        max_idx = int(region['AscendingManifold'])

        min_cell_id = min_idx_to_cell.get(min_idx)
        max_cell_id = max_idx_to_cell.get(max_idx)
        if min_cell_id is None or max_cell_id is None:
            continue

        boundary_saddles = []
        for saddle_cell_id, saddle_idx in saddle_cell_to_idx.items():
            if (min_cell_id in saddle_to_mins.get(saddle_cell_id, set()) and
                    max_cell_id in saddle_to_maxs.get(saddle_cell_id, set())):
                boundary_saddles.append(saddle_idx)

        parent_region_size = region_sizes.get(region_id, 0)

        # Split regions with 3+ boundary saddles into 4-CP sub-regions (2D only)
        if split_large and len(boundary_saddles) >= 3:
            sorted_saddles = _sort_saddles_angularly(
                boundary_saddles,
                min_positions[min_idx],
                max_positions[max_idx],
                saddle_positions,
            )
            n_sorted = len(sorted_saddles)
            n_sub = n_sorted  # ring topology: d saddles → d sub-regions
            sub_region_size = max(1, parent_region_size // n_sub)

            for k in range(n_sub):
                sub_saddles = [sorted_saddles[k], sorted_saddles[(k + 1) % n_sorted]]
                hyperedge = ([min_idx] +
                             [s + saddle_offset for s in sub_saddles] +
                             [max_idx + max_offset])
                sub_id_counter += 1
                rows.append({
                    'region_id': sub_id_counter,
                    'min_id': min_idx,
                    'max_id': max_idx,
                    'num_saddles': 2,
                    'boundary_saddles': str(sub_saddles),
                    'hyperedge': str(hyperedge),
                    'hyperedge_size': 4,
                    'region_size': sub_region_size,
                })
        else:
            boundary_saddles = sorted(boundary_saddles)

            # Global indexing: CPs ordered by CellDimension
            hyperedge = ([min_idx] +
                         [s + saddle_offset for s in boundary_saddles] +
                         [max_idx + max_offset])
            sub_id_counter += 1
            rows.append({
                'region_id': sub_id_counter,
                'min_id': min_idx,
                'max_id': max_idx,
                'num_saddles': len(boundary_saddles),
                'boundary_saddles': str(boundary_saddles),
                'hyperedge': str(hyperedge),
                'hyperedge_size': len(hyperedge),
                'region_size': parent_region_size,
            })

    return pd.DataFrame(rows) if rows else None
