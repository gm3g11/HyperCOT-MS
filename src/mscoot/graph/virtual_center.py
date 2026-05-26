"""Virtual center computation for hyperedges."""

import numpy as np
from typing import List


def compute_virtual_center(coords: np.ndarray, types: np.ndarray,
                           members: List[int]) -> np.ndarray:
    """Compute virtual center for a hyperedge.

    For standard size-4 hyperedges (1 min, 2 saddles, 1 max):
        Line intersection of (min->max) and (saddle1->saddle2).
    For other sizes: centroid of boundary CPs.

    Args:
        coords: All CP coordinates (n_cp, 3).
        types: All CP types (n_cp,).
        members: List of CP indices in this hyperedge.

    Returns:
        Virtual center coordinates (3,).
    """
    member_types = [int(types[m]) for m in members]
    member_coords = np.array([coords[m] for m in members])
    centroid = member_coords.mean(axis=0)

    mins_in = [m for m, t in zip(members, member_types) if t == 0]
    sads_in = [m for m, t in zip(members, member_types) if t == 1]
    # "max" is dim=2 in standard 2D; in 3D dropped-level scheme, the region
    # center is also dim=2 (a 2-saddle), so this handles both cases.
    maxs_in = [m for m, t in zip(members, member_types) if t == 2]

    # For 2D data (z=0): use line intersection for precision
    # For 3D surface data: use centroid (line intersection produces off-surface points)
    is_3d = member_coords.shape[1] >= 3 and np.abs(member_coords[:, 2]).max() > 1e-10

    if not is_3d and len(mins_in) == 1 and len(sads_in) == 2 and len(maxs_in) == 1:
        min_pos = coords[mins_in[0]]
        max_pos = coords[maxs_in[0]]
        s1_pos = coords[sads_in[0]]
        s2_pos = coords[sads_in[1]]

        # 2D line intersection (with z interpolation)
        d1 = max_pos[:2] - min_pos[:2]
        d2 = s2_pos[:2] - s1_pos[:2]

        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        b = np.array([s1_pos[0] - min_pos[0], s1_pos[1] - min_pos[1]])

        det = np.linalg.det(A)
        if abs(det) > 1e-10:
            params = np.linalg.solve(A, b)
            t_val = params[0]
            center_xy = min_pos[:2] + t_val * d1
            z1 = min_pos[2] + t_val * (max_pos[2] - min_pos[2])
            s_val = params[1]
            z2 = s1_pos[2] + s_val * (s2_pos[2] - s1_pos[2])
            center_z = (z1 + z2) / 2
            candidate = np.array([center_xy[0], center_xy[1], center_z])

            # Sanity: candidate must be within member CP bounding box
            # (with 10% margin).  When lines are nearly parallel the
            # intersection shoots far outside the domain — fall back to
            # centroid in that case.
            bbox_min = member_coords[:, :2].min(axis=0)
            bbox_max = member_coords[:, :2].max(axis=0)
            margin = 0.1 * np.linalg.norm(bbox_max - bbox_min)
            if (np.all(candidate[:2] >= bbox_min - margin) and
                    np.all(candidate[:2] <= bbox_max + margin)):
                return candidate

    return centroid
