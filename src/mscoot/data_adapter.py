"""Dataset-agnostic data loading layer.

Handles differences between datasets (column names, file formats,
temporal vs non-temporal, duplicate CellIds) behind a unified interface.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import DatasetConfig, load_dataset_config


class DatasetAdapter:
    """Unified data loading interface for any dataset."""

    def __init__(self, dataset_dir: Path):
        self.config = load_dataset_config(dataset_dir)
        self._cp_cache = None
        self._sep_cache = None
        self._seg_cache = None

    @property
    def name(self) -> str:
        return self.config.name

    def get_timesteps(self) -> List[int]:
        """Get sorted list of available timesteps."""
        if not self.config.is_temporal:
            return list(range(len(self.config.files.get('prefixes', ['0']))))

        ts_col = self.config.columns.get('timestep', 'TimeStep')
        cp_df = self._load_raw_cp()
        return sorted(cp_df[ts_col].unique().astype(int))

    def load_critical_points(self, timestep=None) -> pd.DataFrame:
        """Load critical points with standardized column names.

        Returns DataFrame with columns: x, y, z, cp_type, cell_id, scalar
        """
        raw = self._load_raw_cp()
        cols = self.config.columns

        if self.config.is_temporal and timestep is not None:
            ts_col = cols.get('timestep', 'TimeStep')
            raw = raw[raw[ts_col] == timestep].copy()

        if not self.config.is_temporal and isinstance(timestep, (int, str)):
            # Non-temporal: timestep is a prefix index or name
            prefixes = self.config.files.get('prefixes', [])
            if isinstance(timestep, int) and timestep < len(prefixes):
                prefix = prefixes[timestep]
            else:
                prefix = str(timestep)
            pattern = self.config.files['cp_pattern'].format(prefix=prefix)
            raw = pd.read_csv(self.config.ttk_output_dir / pattern)

        result = pd.DataFrame({
            'x': raw[cols['x']].values,
            'y': raw[cols['y']].values,
            'z': raw[cols.get('z', cols['y'])].values if 'z' in cols else 0.0,
            'cp_type': raw[cols['cp_type']].values.astype(int),
            'cell_id': raw[cols['cell_id']].values.astype(int),
            'scalar': raw[cols['scalar']].values,
        })

        return result.reset_index(drop=True)

    def load_edges(self, timestep=None) -> List[Tuple[int, int]]:
        """Load separatrix edges as (local_idx_i, local_idx_j) pairs.

        Requires CPs for the same timestep to build CellId -> local index mapping.
        Handles duplicate CellIds correctly.
        Uses vectorized operations for large datasets.
        """
        cp_df = self.load_critical_points(timestep)
        sep_df = self._load_raw_sep(timestep)
        sep_cfg = self.config.separatrices

        src_col = sep_cfg.get('source', 'SourceId')
        dst_col = sep_cfg.get('destination', 'DestinationId')
        dedup_cols = sep_cfg.get('dedup_on', [src_col, dst_col])

        # Build CellId -> local index mapping (handle duplicates)
        cell_id_to_idx = defaultdict(list)
        cp_types = cp_df['cp_type'].values
        cell_ids = cp_df['cell_id'].values
        for idx, cid in enumerate(cell_ids):
            cell_id_to_idx[int(cid)].append(idx)

        # Build CellId+type -> index for disambiguation
        cellid_type_to_idx = {}
        for idx, (cid, ctype) in enumerate(zip(cell_ids, cp_types)):
            cellid_type_to_idx[(int(cid), int(ctype))] = idx

        # Deduplicate separatrices
        if len(sep_df) > 0 and dedup_cols:
            available_cols = [c for c in dedup_cols if c in sep_df.columns]
            if available_cols:
                sep_df = sep_df.drop_duplicates(subset=available_cols)

        if len(sep_df) == 0:
            return []

        sep_type_col = sep_cfg.get('sep_type', 'SeparatrixType')
        has_sep_type = sep_type_col in sep_df.columns

        # Check which CellIds have duplicates (rare — TTK bug)
        has_duplicates = any(len(v) > 1 for v in cell_id_to_idx.values())

        # --- Fast vectorized path (no duplicate CellIds or has SeparatrixType) ---
        if has_sep_type and (not has_duplicates or True):
            # Build simple CellId -> single index (first match)
            cellid_to_single = {int(cid): idxs[0] for cid, idxs in cell_id_to_idx.items()}
            valid_cids = set(cellid_to_single.keys())

            src_arr = sep_df[src_col].values.astype(int)
            dst_arr = sep_df[dst_col].values.astype(int)
            stype_arr = sep_df[sep_type_col].values.astype(int)

            edges_set = set()
            # Vectorized: filter to rows where both src and dst are known CPs
            import numpy as np
            src_valid = np.array([s in valid_cids for s in src_arr])
            dst_valid = np.array([d in valid_cids for d in dst_arr])
            both_valid = src_valid & dst_valid

            src_filt = src_arr[both_valid]
            dst_filt = dst_arr[both_valid]
            stype_filt = stype_arr[both_valid]

            # Process by separatrix type (vectorized lookup)
            for stype, src_dim, dst_dim in [(0, 1, 0), (1, 1, 2), (2, 2, 3)]:
                mask = stype_filt == stype
                if not mask.any():
                    continue
                srcs = src_filt[mask]
                dsts = dst_filt[mask]
                for s, d in zip(srcs, dsts):
                    si = cellid_type_to_idx.get((int(s), src_dim))
                    di = cellid_type_to_idx.get((int(d), dst_dim))
                    if si is not None and di is not None and si != di:
                        edges_set.add((min(si, di), max(si, di)))

            return list(edges_set)

        # --- Slow fallback path (row-by-row, for edge cases) ---
        edges = []
        for _, row in sep_df.iterrows():
            src = int(row[src_col])
            dst = int(row[dst_col])
            if src not in cell_id_to_idx or dst not in cell_id_to_idx:
                continue

            src_indices = cell_id_to_idx[src]
            dst_indices = cell_id_to_idx[dst]

            if len(src_indices) == 1 and len(dst_indices) == 1:
                if src_indices[0] != dst_indices[0]:
                    edges.append((src_indices[0], dst_indices[0]))
                continue

            if has_sep_type:
                stype = int(row[sep_type_col])
                if stype == 0:
                    src_i = cellid_type_to_idx.get((src, 1))
                    dst_i = cellid_type_to_idx.get((dst, 0))
                elif stype == 2:
                    src_i = cellid_type_to_idx.get((src, 2))
                    dst_i = cellid_type_to_idx.get((dst, 3))
                else:
                    src_i = cellid_type_to_idx.get((src, 1))
                    dst_i = cellid_type_to_idx.get((dst, 2))
                if src_i is not None and dst_i is not None and src_i != dst_i:
                    edges.append((src_i, dst_i))
            else:
                for i in src_indices:
                    for j in dst_indices:
                        if i != j:
                            edges.append((i, j))

        # Deduplicate undirected edges
        return list(set((min(a, b), max(a, b)) for a, b in edges))

    def load_segmentation(self, timestep=None) -> pd.DataFrame:
        """Load segmentation data for a timestep."""
        raw = self._load_raw_seg()
        if self.config.is_temporal and timestep is not None:
            ts_col = self.config.columns.get('timestep', 'TimeStep')
            raw = raw[raw[ts_col] == timestep].copy()
        return raw.reset_index(drop=True)

    def get_domain_diagonal(self) -> float:
        """Get domain diagonal distance for normalization."""
        if self.config.diagonal is not None:
            return self.config.diagonal

        extent = self.config.domain.get('extent')
        if extent:
            sq_sum = sum((e[1] - e[0]) ** 2 for e in extent)
            return np.sqrt(sq_sum)

        # Compute from data
        cp_df = self._load_raw_cp()
        cols = self.config.columns
        coord_cols = [cols['x'], cols['y']]
        if self.config.n_dimensions >= 3 and 'z' in cols:
            coord_cols.append(cols['z'])
        coords = cp_df[coord_cols].values
        extent_diag = np.sqrt(
            sum((coords[:, i].max() - coords[:, i].min()) ** 2
                for i in range(coords.shape[1]))
        )
        return extent_diag

    def build_cellid_mapping(self, cp_data: pd.DataFrame) -> Dict[int, List[int]]:
        """Build CellId -> list of local indices mapping."""
        mapping = defaultdict(list)
        for idx, cid in enumerate(cp_data['cell_id'].values):
            mapping[int(cid)].append(idx)
        return dict(mapping)

    # --- Private helpers ---

    def _load_raw_cp(self) -> pd.DataFrame:
        if self._cp_cache is not None:
            return self._cp_cache

        if self.config.is_temporal:
            pattern = self.config.files['cp_pattern']
            path = self.config.ttk_output_dir / pattern
            self._cp_cache = pd.read_csv(path)
        else:
            # Non-temporal: load all prefixes and cache
            frames = []
            for i, prefix in enumerate(self.config.files.get('prefixes', [])):
                pattern = self.config.files['cp_pattern'].format(prefix=prefix)
                df = pd.read_csv(self.config.ttk_output_dir / pattern)
                df['_prefix_idx'] = i
                df['_prefix'] = prefix
                frames.append(df)
            self._cp_cache = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        return self._cp_cache

    def _load_raw_sep(self, timestep=None) -> pd.DataFrame:
        if self._sep_cache is None:
            if self.config.is_temporal:
                pattern = self.config.files.get('sep_pattern', 'separatrices.csv')
                path = self.config.ttk_output_dir / pattern
                if path.exists():
                    self._sep_cache = pd.read_csv(path)
                else:
                    self._sep_cache = pd.DataFrame()
            else:
                frames = []
                for prefix in self.config.files.get('prefixes', []):
                    pattern = self.config.files['sep_pattern'].format(prefix=prefix)
                    path = self.config.ttk_output_dir / pattern
                    if path.exists():
                        df = pd.read_csv(path)
                        df['_prefix'] = prefix
                        frames.append(df)
                self._sep_cache = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        sep = self._sep_cache
        if self.config.is_temporal and timestep is not None:
            ts_col = self.config.columns.get('timestep', 'TimeStep')
            if ts_col in sep.columns:
                sep = sep[sep[ts_col] == timestep]
        return sep

    def _load_raw_seg(self) -> pd.DataFrame:
        if self._seg_cache is not None:
            return self._seg_cache

        if self.config.is_temporal:
            pattern = self.config.files.get('seg_pattern', 'segmentation.csv')
            path = self.config.ttk_output_dir / pattern
            if path.exists():
                self._seg_cache = pd.read_csv(path)
            else:
                self._seg_cache = pd.DataFrame()
        else:
            frames = []
            for prefix in self.config.files.get('prefixes', []):
                pattern = self.config.files['seg_pattern'].format(prefix=prefix)
                path = self.config.ttk_output_dir / pattern
                if path.exists():
                    df = pd.read_csv(path)
                    df['_prefix'] = prefix
                    frames.append(df)
            self._seg_cache = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        return self._seg_cache
