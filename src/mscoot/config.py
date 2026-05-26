"""Dataset configuration loading and parameter defaults."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# Default algorithm parameters
DEFAULTS = {
    'sigma': 0.3,
    'epsilon': 0.01,
    'alpha': 0.5,
    'mu_floor': 0.20,
    'persistence_threshold': 0.01,
    'pi_resolution': 100,
}


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    dataset_dir: Path
    is_temporal: bool
    n_dimensions: int
    columns: Dict[str, str]
    separatrices: Dict[str, str]
    files: Dict[str, str]
    domain: Dict
    parameters: Dict[str, float]

    @property
    def ttk_output_dir(self) -> Path:
        return self.dataset_dir / "ttk_output"

    @property
    def mesh_dir(self) -> Path:
        return self.dataset_dir / "mesh"

    @property
    def has_duplicate_cellids(self) -> bool:
        return self.domain.get('has_duplicate_cellids', False)

    @property
    def diagonal(self) -> Optional[float]:
        return self.domain.get('diagonal')

    def get_param(self, key: str) -> float:
        return self.parameters.get(key, DEFAULTS.get(key))


def load_dataset_config(dataset_dir: Path) -> DatasetConfig:
    """Load dataset.yaml from a dataset directory."""
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No dataset.yaml found in {dataset_dir}")

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    return DatasetConfig(
        name=raw['name'],
        dataset_dir=dataset_dir,
        is_temporal=raw.get('is_temporal', True),
        n_dimensions=raw.get('n_dimensions', 2),
        columns=raw.get('columns', {}),
        separatrices=raw.get('separatrices', {}),
        files=raw.get('files', {}),
        domain=raw.get('domain', {}),
        parameters=raw.get('parameters', {}),
    )


def find_project_root() -> Path:
    """Find the project root (directory containing pyproject.toml)."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml)")


def get_dataset_dir(dataset_name: str) -> Path:
    """Get the dataset directory for a named dataset."""
    return find_project_root() / "datasets" / dataset_name


def get_results_dir(dataset_name: str) -> Path:
    """Get the results directory for a named dataset."""
    return find_project_root() / "results" / dataset_name
