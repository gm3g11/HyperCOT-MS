#!/usr/bin/env python3
"""Convert TOSCA .vert/.tri meshes to VTP with average geodesic distance scalar field.

Scalar function: f(v) = mean geodesic distance from K farthest-point-sampled anchors.
This is the standard isometry-invariant scalar for shape comparison (Sridharamurthy 2020,
Bollen 2022, Yan 2022, Pont 2021, Lyu/LSH 2024, Rathore/MTNN 2024).

Geodesic algorithm: Heat method via potpourri3d (geometry-central C++ backend).
<1% error vs exact geodesic, well below persistence threshold.

Usage:
    python scripts/prepare_tosca.py
    python scripts/prepare_tosca.py --validate   # also run exact geodesic comparison
    python scripts/prepare_tosca.py --anchors 15 # change anchor count (default: 15)
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

# VTK imports for writing VTP
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def natural_sort_key(name):
    """Sort key for natural ordering: cat0, cat1, ..., cat10, centaur0, ..."""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', name)]


def read_vert_tri(vert_path, tri_path):
    """Read .vert and .tri files.

    Returns:
        verts: (N, 3) float64 array
        tris: (M, 3) int32 array, 0-indexed
    """
    verts = np.loadtxt(vert_path, dtype=np.float64)
    tris = np.loadtxt(tri_path, dtype=np.int32)

    # TOSCA .tri files are 1-indexed → convert to 0-indexed
    if tris.min() >= 1:
        tris = tris - 1

    return verts, tris


def clean_mesh(verts, tris):
    """Remove unreferenced vertices and reindex triangles.

    Some TOSCA meshes have vertices not referenced by any triangle,
    which causes geometry-central to fail.

    Returns:
        clean_verts: (N', 3) vertices with unreferenced removed
        clean_tris: (M, 3) triangles with reindexed vertex ids
        old_to_new: (N,) mapping from old to new vertex indices (-1 if removed)
    """
    n_verts = len(verts)
    referenced = np.zeros(n_verts, dtype=bool)
    referenced[tris.ravel()] = True

    if referenced.all():
        return verts, tris, np.arange(n_verts)

    # Build reindex mapping
    old_to_new = np.full(n_verts, -1, dtype=int)
    new_idx = 0
    for i in range(n_verts):
        if referenced[i]:
            old_to_new[i] = new_idx
            new_idx += 1

    clean_verts = verts[referenced]
    clean_tris = old_to_new[tris]

    return clean_verts, clean_tris, old_to_new


def _find_components(n_verts, tris):
    """Find connected components of a mesh. Returns component label per vertex."""
    from collections import deque
    adj = [set() for _ in range(n_verts)]
    for tri in tris:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[tri[i]].add(tri[j])
    labels = np.full(n_verts, -1, dtype=int)
    comp_id = 0
    for start in range(n_verts):
        if labels[start] >= 0:
            continue
        q = deque([start])
        while q:
            v = q.popleft()
            if labels[v] >= 0:
                continue
            labels[v] = comp_id
            for nb in adj[v]:
                if labels[nb] < 0:
                    q.append(nb)
        comp_id += 1
    return labels, comp_id


def _extract_component(verts, tris, mask):
    """Extract a connected component as a sub-mesh.

    Returns:
        sub_verts, sub_tris, old_to_new mapping
    """
    indices = np.where(mask)[0]
    old_to_new = np.full(len(verts), -1, dtype=int)
    for new_i, old_i in enumerate(indices):
        old_to_new[old_i] = new_i
    sub_tris = []
    for tri in tris:
        if mask[tri[0]] and mask[tri[1]] and mask[tri[2]]:
            sub_tris.append([old_to_new[v] for v in tri])
    return verts[mask], np.array(sub_tris, dtype=np.int32), old_to_new


def _compute_agd_single_component(verts, tris, n_anchors):
    """Compute AGD on a single connected component via heat method + FPS."""
    import potpourri3d as pp3d
    solver = pp3d.MeshHeatMethodDistanceSolver(verts, tris)
    anchors = [0]
    d0 = solver.compute_distance(0)
    min_dist = d0.copy()
    all_dists = [d0]
    for k in range(1, n_anchors):
        next_a = np.argmax(min_dist)
        anchors.append(int(next_a))
        dk = solver.compute_distance(next_a)
        min_dist = np.minimum(min_dist, dk)
        all_dists.append(dk)
    return np.mean(all_dists, axis=0), anchors


def extract_largest_component(verts, tris):
    """Extract the largest connected component of a mesh.

    The TOSCA gorilla meshes have 38 disconnected components (main body +
    37 hair/fur fragments on the head). The fragments are tiny (187-2644 verts,
    bbox 1-12) compared to the main body (25,438 verts, bbox 255). Keeping
    them corrupts the AGD scalar field because the heat method assigns a
    constant distance to cross-component vertices, creating thousands of
    false extrema.

    Returns:
        (verts, tris) of the largest component, or the original mesh if
        already connected. Also returns n_components found.
    """
    # Clean unreferenced vertices first
    clean_verts, clean_tris, _ = clean_mesh(verts, tris)
    comp_labels, n_comps = _find_components(len(clean_verts), clean_tris)

    if n_comps == 1:
        return clean_verts, clean_tris, 1

    comp_sizes = np.bincount(comp_labels)
    largest = np.argmax(comp_sizes)
    mask = comp_labels == largest

    lc_verts, lc_tris, _ = _extract_component(clean_verts, clean_tris, mask)
    return lc_verts, lc_tris, n_comps


def compute_agd(verts, tris, n_anchors=15):
    """Compute Average Geodesic Distance via heat method + FPS.

    For disconnected meshes, uses only the largest connected component
    (discards small fragments like gorilla hair/fur).

    Args:
        verts: (N, 3) vertices
        tris: (M, 3) triangles (0-indexed)
        n_anchors: number of FPS anchor points

    Returns:
        agd: (N,) average geodesic distance, normalized to [0, 1]
        anchor_indices: list of anchor vertex indices
        verts_used: (N', 3) vertices actually used (may be subset for disconnected meshes)
        tris_used: (M', 3) triangles actually used
        n_components: number of connected components found
    """
    verts_used, tris_used, n_components = extract_largest_component(verts, tris)

    agd_raw, anchor_indices = _compute_agd_single_component(
        verts_used, tris_used, n_anchors)

    # Normalize to [0, 1]
    agd_min, agd_max = agd_raw.min(), agd_raw.max()
    if agd_max - agd_min > 1e-10:
        agd = (agd_raw - agd_min) / (agd_max - agd_min)
    else:
        agd = np.zeros_like(agd_raw)

    return agd, anchor_indices, verts_used, tris_used, n_components


def write_vtp(verts, tris, agd, out_path):
    """Write mesh with AGD scalar field as VTP."""
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(verts, deep=True))

    cells = vtk.vtkCellArray()
    for tri in tris:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, int(tri[0]))
        triangle.GetPointIds().SetId(1, int(tri[1]))
        triangle.GetPointIds().SetId(2, int(tri[2]))
        cells.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    # Add AGD as point scalar
    agd_vtk = numpy_to_vtk(agd, deep=True)
    agd_vtk.SetName("avg_geodesic")
    polydata.GetPointData().AddArray(agd_vtk)
    polydata.GetPointData().SetActiveScalars("avg_geodesic")

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(polydata)
    writer.Write()


def count_local_extrema(agd, tris):
    """Count local extrema (minima + maxima) of AGD field on mesh."""
    n_verts = len(agd)

    # Build vertex adjacency
    neighbors = [set() for _ in range(n_verts)]
    for tri in tris:
        for i in range(3):
            for j in range(3):
                if i != j:
                    neighbors[tri[i]].add(tri[j])

    n_min = 0
    n_max = 0
    for v in range(n_verts):
        vals = [agd[n] for n in neighbors[v]]
        if not vals:
            continue
        if agd[v] <= min(vals):
            n_min += 1
        if agd[v] >= max(vals):
            n_max += 1

    return n_min, n_max


def validate_geodesics(verts, tris, mesh_name, n_anchors=15):
    """Validate heat method vs exact geodesic (MMP) on a few anchors.

    Returns dict with comparison statistics, or None if pygeodesic unavailable.
    """
    try:
        import pygeodesic.geodesic as geodesic
    except ImportError:
        print(f"  [validate] pygeodesic not available, skipping validation for {mesh_name}")
        return None

    import potpourri3d as pp3d

    # Use largest connected component for both solvers
    clean_verts, clean_tris, n_comps = extract_largest_component(verts, tris)
    if n_comps > 1:
        print(f"  [validate] {mesh_name}: using largest component ({len(clean_verts)} verts, {n_comps} total components)")

    # Heat method
    heat_solver = pp3d.MeshHeatMethodDistanceSolver(clean_verts, clean_tris)

    # Exact geodesic (MMP)
    geoalg = geodesic.PyGeodesicAlgorithmExact(clean_verts, clean_tris)

    # Compare on first 3 FPS anchors
    anchors = [0]
    d0_heat = heat_solver.compute_distance(0)
    min_dist = d0_heat.copy()

    for k in range(1, min(3, n_anchors)):
        next_a = np.argmax(min_dist)
        anchors.append(int(next_a))
        dk = heat_solver.compute_distance(next_a)
        min_dist = np.minimum(min_dist, dk)

    rel_errors = []
    for anchor in anchors:
        d_heat = heat_solver.compute_distance(anchor)
        d_exact, _ = geoalg.geodesicDistances(np.array([anchor]), None)

        # Relative error where exact > 0
        mask = d_exact > 1e-10
        if mask.sum() > 0:
            rel_err = np.abs(d_heat[mask] - d_exact[mask]) / d_exact[mask]
            rel_errors.append(rel_err.mean())

    mean_rel_error = np.mean(rel_errors) if rel_errors else 0.0

    # Check extrema count consistency
    agd_heat, _, v_used, t_used, _ = compute_agd(verts, tris, n_anchors)

    # Count extrema with heat method
    n_min_heat, n_max_heat = count_local_extrema(agd_heat, t_used)

    result = {
        'mesh': mesh_name,
        'mean_rel_error': mean_rel_error,
        'n_anchors_tested': len(anchors),
        'n_min_heat': n_min_heat,
        'n_max_heat': n_max_heat,
        'n_extrema_heat': n_min_heat + n_max_heat,
    }

    print(f"  [validate] {mesh_name}: mean_rel_error={mean_rel_error:.4f} "
          f"({mean_rel_error*100:.2f}%), extrema={n_min_heat}min+{n_max_heat}max")

    return result


def get_class_label(name):
    """Extract class name from mesh name (e.g., 'cat0' -> 'cat')."""
    return re.match(r'^([a-z]+)', name).group(1)


def main():
    parser = argparse.ArgumentParser(description='Prepare TOSCA meshes with AGD scalar field')
    parser.add_argument('--anchors', type=int, default=15,
                        help='Number of FPS anchor points (default: 15)')
    parser.add_argument('--validate', action='store_true',
                        help='Run geodesic accuracy validation on representative meshes')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    tosca_dir = project_root / "datasets" / "tosca"
    mesh_dir = tosca_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Find all .vert files
    vert_files = sorted(tosca_dir.glob("*.vert"), key=lambda p: natural_sort_key(p.stem))
    if not vert_files:
        print(f"ERROR: No .vert files found in {tosca_dir}")
        sys.exit(1)

    print(f"Found {len(vert_files)} meshes in {tosca_dir}")
    print(f"Using {args.anchors} FPS anchors")
    print(f"Output: {mesh_dir}/")
    print("=" * 60)

    # Validation on representative meshes
    if args.validate:
        print("\n--- Geodesic validation ---")
        for name in ['cat0', 'michael0', 'wolf0']:
            vp = tosca_dir / f"{name}.vert"
            tp = tosca_dir / f"{name}.tri"
            if vp.exists() and tp.exists():
                verts, tris = read_vert_tri(vp, tp)
                validate_geodesics(verts, tris, name, args.anchors)
        print()

    # Process all meshes
    stats_rows = []
    for i, vert_path in enumerate(vert_files):
        name = vert_path.stem
        tri_path = vert_path.with_suffix('.tri')
        if not tri_path.exists():
            print(f"  [{i+1}/{len(vert_files)}] {name}: SKIPPED (no .tri)")
            continue

        verts_raw, tris_raw = read_vert_tri(vert_path, tri_path)
        agd, anchors, verts, tris, n_comps = compute_agd(
            verts_raw, tris_raw, n_anchors=args.anchors)

        # Count extrema
        n_min, n_max = count_local_extrema(agd, tris)

        # Write VTP (uses cleaned/extracted mesh, not raw)
        out_path = mesh_dir / f"{name}.vtp"
        write_vtp(verts, tris, agd, out_path)

        class_label = get_class_label(name)
        dropped = len(verts_raw) - len(verts)
        stats_rows.append({
            'mesh': name,
            'class': class_label,
            'n_vertices_raw': len(verts_raw),
            'n_vertices': len(verts),
            'n_triangles': len(tris),
            'n_components': n_comps,
            'n_dropped': dropped,
            'agd_min': float(agd.min()),
            'agd_max': float(agd.max()),
            'agd_mean': float(agd.mean()),
            'agd_std': float(agd.std()),
            'n_local_minima': n_min,
            'n_local_maxima': n_max,
        })

        extra = ""
        if n_comps > 1:
            extra = f" [kept largest of {n_comps} components, dropped {dropped:,}]"
        print(f"  [{i+1}/{len(vert_files)}] {name}: "
              f"{len(verts):,} verts, {len(tris):,} tris, "
              f"extrema={n_min}min+{n_max}max, class={class_label}{extra}")

    # Save AGD statistics
    import pandas as pd
    stats_df = pd.DataFrame(stats_rows)
    stats_path = mesh_dir / "agd_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    # Print summary by class
    print(f"\n{'=' * 60}")
    print("Summary by class:")
    for cls in stats_df['class'].unique():
        sub = stats_df[stats_df['class'] == cls]
        print(f"  {cls:10s}: {len(sub):2d} meshes, "
              f"verts={sub['n_vertices'].min():,}-{sub['n_vertices'].max():,}, "
              f"avg extrema={sub['n_local_minima'].mean() + sub['n_local_maxima'].mean():.0f}")

    print(f"\nTotal: {len(stats_rows)} VTP files written to {mesh_dir}/")
    print(f"Stats: {stats_path}")


if __name__ == '__main__':
    main()
