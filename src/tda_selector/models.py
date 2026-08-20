from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunConfig:
    structures: str
    index: str = ":"
    output_dir: Optional[str] = None
    features_cache: Optional[str] = None
    max_edge_length: float = 6.0
    homology_dims: tuple[int, ...] = (0, 1)
    bandwidth: float = 0.5
    resolution: tuple[int, int] = (20, 20)
    per_species: bool = True
    species: Optional[list[str]] = None
    max_atoms_per_species: Optional[int] = None
    supercell: tuple[int, int, int] = (1, 1, 1)
    n_jobs: int = -1
    selection_method: str = "fps"
    n_select: Optional[int] = None
    min_distance: Optional[float] = None
    start: str = "centroid"
    no_fill: bool = False
    plot: bool = False
    seed: int = 42


@dataclass
class RunResult:
    selected_indices: list[int]
    output_dir: Path
    summary_path: Path
    selected_structures_path: Path
    indices_path: Path
    stopping_condition: str
