"""TTK-based Morse-Smale complex extraction from VTI/VTP meshes.

Requires vtk and topologytoolkit Python packages (optional dependency).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def extract_ms_complex(mesh_path: Path, persistence_threshold: float = 0.01,
                       field_name: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Extract critical points, separatrices, and segmentation from a mesh.

    Pipeline:
    1. Read mesh (VTI/VTP) with appropriate VTK reader
    2. Compute persistence diagram (ttkPersistenceDiagram)
    3. Simplify by persistence threshold (ttkTopologicalSimplification)
    4. Compute Morse-Smale complex (ttkMorseSmaleComplex)
    5. Extract CPs, separatrices, segmentation as DataFrames

    Args:
        mesh_path: Path to VTI or VTP mesh file.
        persistence_threshold: Simplification threshold (fraction of scalar range).
        field_name: Name of scalar field to use. Auto-detected if None.

    Returns:
        Dict with keys 'critical_points', 'separatrices', 'segmentation',
        each containing a pandas DataFrame.

    Raises:
        ImportError: If VTK or TTK are not installed.
    """
    try:
        import vtk
        import topologytoolkit as ttk
    except ImportError as e:
        raise ImportError(
            "TTK extraction requires vtk and topologytoolkit packages. "
            "Install with: pip install vtk topologytoolkit"
        ) from e

    mesh_path = Path(mesh_path)
    ext = mesh_path.suffix.lower()

    # 1. Read mesh
    if ext == '.vti':
        reader = vtk.vtkXMLImageDataReader()
    elif ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")

    reader.SetFileName(str(mesh_path))
    reader.Update()
    mesh = reader.GetOutput()

    # Auto-detect scalar field
    if field_name is None:
        pd_data = mesh.GetPointData()
        if pd_data.GetNumberOfArrays() > 0:
            field_name = pd_data.GetArrayName(0)
        else:
            raise ValueError("No point data arrays found in mesh")

    mesh.GetPointData().SetActiveScalars(field_name)

    # 2. Persistence diagram
    persistence = ttk.ttkPersistenceDiagram()
    persistence.SetInputData(mesh)
    persistence.Update()

    # 3. Topological simplification
    # Get scalar range for threshold computation
    scalar_range = mesh.GetPointData().GetScalars().GetRange()
    abs_threshold = persistence_threshold * (scalar_range[1] - scalar_range[0])

    simplification = ttk.ttkTopologicalSimplification()
    simplification.SetInputData(0, mesh)
    simplification.SetInputData(1, persistence.GetOutput())
    simplification.SetPersistenceThreshold(abs_threshold)
    simplification.Update()

    # 4. Morse-Smale complex
    ms_complex = ttk.ttkMorseSmaleComplex()
    ms_complex.SetInputData(simplification.GetOutput())
    ms_complex.Update()

    # 5. Extract DataFrames
    result = {}

    # Critical points (output port 0)
    cp_data = ms_complex.GetOutput(0)
    result['critical_points'] = _vtk_to_cp_dataframe(cp_data)

    # Separatrices (output port 1)
    sep_data = ms_complex.GetOutput(1)
    result['separatrices'] = _vtk_to_sep_dataframe(sep_data)

    # Segmentation (output port 3)
    seg_data = ms_complex.GetOutput(3)
    result['segmentation'] = _vtk_to_seg_dataframe(seg_data)

    return result


def _vtk_to_cp_dataframe(vtk_data) -> pd.DataFrame:
    """Convert VTK critical points to DataFrame."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    n_points = vtk_data.GetNumberOfPoints()
    points = vtk_to_numpy(vtk_data.GetPoints().GetData())

    row_data = {
        'Points:0': points[:, 0],
        'Points:1': points[:, 1],
        'Points:2': points[:, 2],
    }

    pd_data = vtk_data.GetPointData()
    for i in range(pd_data.GetNumberOfArrays()):
        name = pd_data.GetArrayName(i)
        arr = vtk_to_numpy(pd_data.GetArray(i))
        row_data[name] = arr

    return pd.DataFrame(row_data)


def _vtk_to_sep_dataframe(vtk_data) -> pd.DataFrame:
    """Convert VTK separatrices to DataFrame."""
    from vtk.util.numpy_support import vtk_to_numpy

    n_cells = vtk_data.GetNumberOfCells()
    rows = []

    cd = vtk_data.GetCellData()
    field_names = [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())]

    for i in range(n_cells):
        cell = vtk_data.GetCell(i)
        if cell.GetNumberOfPoints() == 2:
            row = {}
            for fname in field_names:
                arr = cd.GetArray(fname)
                row[fname] = arr.GetTuple1(i) if arr.GetNumberOfComponents() == 1 else arr.GetTuple(i)
            rows.append(row)

    return pd.DataFrame(rows)


def _vtk_to_seg_dataframe(vtk_data) -> pd.DataFrame:
    """Convert VTK segmentation to DataFrame."""
    from vtk.util.numpy_support import vtk_to_numpy

    pd_data = vtk_data.GetPointData()
    row_data = {}

    for i in range(pd_data.GetNumberOfArrays()):
        name = pd_data.GetArrayName(i)
        arr = vtk_to_numpy(pd_data.GetArray(i))
        row_data[name] = arr

    return pd.DataFrame(row_data)


def extract_dataset(dataset_dir: Path, persistence_threshold: float = 0.01,
                    field_name: Optional[str] = None):
    """Extract MS complex for all mesh files in a dataset.

    Saves results to dataset_dir/ttk_output/ as CSVs.
    """
    mesh_dir = dataset_dir / "mesh"
    output_dir = dataset_dir / "ttk_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = sorted(mesh_dir.glob("*.vt[ip]"))
    if not mesh_files:
        raise FileNotFoundError(f"No VTI/VTP files found in {mesh_dir}")

    print(f"Extracting MS complex for {len(mesh_files)} mesh files...")

    all_cps = []
    all_seps = []
    all_segs = []

    for i, mesh_path in enumerate(mesh_files):
        print(f"  [{i+1}/{len(mesh_files)}] {mesh_path.name}...")
        result = extract_ms_complex(mesh_path, persistence_threshold, field_name)

        for key in result:
            result[key]['TimeStep'] = i

        all_cps.append(result['critical_points'])
        all_seps.append(result['separatrices'])
        all_segs.append(result['segmentation'])

    # Save combined CSVs
    pd.concat(all_cps, ignore_index=True).to_csv(
        output_dir / "critical_points.csv", index=False
    )
    pd.concat(all_seps, ignore_index=True).to_csv(
        output_dir / "separatrices.csv", index=False
    )
    pd.concat(all_segs, ignore_index=True).to_csv(
        output_dir / "segmentation.csv", index=False
    )

    # Save threshold info
    (output_dir / "threshold_used.txt").write_text(
        f"persistence_threshold={persistence_threshold}\n"
    )

    print(f"Saved TTK output to {output_dir}/")
