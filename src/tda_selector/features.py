from typing import Dict, Optional

import ase.io
import gudhi
import numpy as np
from ase import Atoms
from gudhi.representations import PersistenceImage
from tqdm import tqdm

try:
    from joblib import Parallel, delayed

    _JOBLIB = True
except ImportError:
    _JOBLIB = False


def load_structures(path: str, index: str = ":") -> list[Atoms]:
    structures = ase.io.read(path, index=index)
    if isinstance(structures, Atoms):
        structures = [structures]
    if not structures:
        raise ValueError("No readable frames found")
    return structures


def _rips_persistence(positions: np.ndarray, max_edge_length: float, max_dim: int) -> Dict[int, np.ndarray]:
    if len(positions) < 2:
        return {d: np.empty((0, 2)) for d in range(max_dim)}

    rips = gudhi.RipsComplex(points=positions, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    st.compute_persistence()

    diagrams: Dict[int, np.ndarray] = {}
    for dim in range(max_dim):
        pairs = np.array(st.persistence_intervals_in_dimension(dim))
        diagrams[dim] = pairs[np.isfinite(pairs[:, 1])] if pairs.size else np.empty((0, 2))
    return diagrams


def _diagrams_for_structure(
    atoms: Atoms,
    max_edge_length: float,
    max_dim: int,
    per_species: bool,
    max_atoms_per_species: Optional[int],
    species_filter: Optional[list[str]],
    supercell: tuple[int, int, int],
) -> Dict[str, Dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed=42)

    if supercell != (1, 1, 1):
        atoms = atoms.repeat(list(supercell))

    symbols = np.array(atoms.get_chemical_symbols())
    positions_all = atoms.positions

    if species_filter is not None:
        mask = np.isin(symbols, species_filter)
        symbols = symbols[mask]
        positions_all = positions_all[mask]
        if len(positions_all) == 0:
            raise ValueError("Species filter removed all atoms")

    def maybe_subsample(pos: np.ndarray) -> np.ndarray:
        if max_atoms_per_species and len(pos) > max_atoms_per_species:
            idx = rng.choice(len(pos), max_atoms_per_species, replace=False)
            return pos[idx]
        return pos

    if per_species:
        out: Dict[str, Dict[int, np.ndarray]] = {}
        for sp in np.unique(symbols):
            out[sp] = _rips_persistence(maybe_subsample(positions_all[symbols == sp]), max_edge_length, max_dim)
        return out

    return {"all": _rips_persistence(maybe_subsample(positions_all), max_edge_length, max_dim)}


def compute_all_diagrams(
    structures: list[Atoms],
    max_edge_length: float = 6.0,
    max_dim: int = 2,
    per_species: bool = True,
    max_atoms_per_species: Optional[int] = None,
    species_filter: Optional[list[str]] = None,
    supercell: tuple[int, int, int] = (1, 1, 1),
    n_jobs: int = -1,
) -> list[Dict[str, Dict[int, np.ndarray]]]:
    if _JOBLIB and n_jobs != 1:
        fn = delayed(_diagrams_for_structure)
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            fn(a, max_edge_length, max_dim, per_species, max_atoms_per_species, species_filter, supercell)
            for a in tqdm(structures, desc="Computing persistence diagrams")
        )

    return [
        _diagrams_for_structure(a, max_edge_length, max_dim, per_species, max_atoms_per_species, species_filter, supercell)
        for a in tqdm(structures, desc="Computing persistence diagrams")
    ]


def _fit_pi(diagrams_list: list[np.ndarray], bandwidth: float, resolution: tuple[int, int]) -> np.ndarray:
    non_empty = [d for d in diagrams_list if len(d) > 0]
    n = len(diagrams_list)
    feat_dim = resolution[0] * resolution[1]
    if not non_empty:
        return np.zeros((n, feat_dim))

    pi = PersistenceImage(bandwidth=bandwidth, resolution=list(resolution))
    pi.fit(non_empty)
    vectors = np.zeros((n, feat_dim))
    for i, diag in enumerate(diagrams_list):
        if len(diag) > 0:
            vectors[i] = pi.transform([diag])[0]
    return vectors


def diagrams_to_features(
    all_diagrams: list[Dict[str, Dict[int, np.ndarray]]],
    homology_dims: tuple[int, ...] = (0, 1),
    bandwidth: float = 0.5,
    resolution: tuple[int, int] = (20, 20),
) -> np.ndarray:
    all_species = sorted({sp for d in all_diagrams for sp in d})
    blocks: list[np.ndarray] = []

    for sp in all_species:
        for dim in homology_dims:
            diagrams_list = [d[sp][dim] if sp in d and dim in d[sp] else np.empty((0, 2)) for d in all_diagrams]
            vec = _fit_pi(diagrams_list, bandwidth, resolution)
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            blocks.append(vec / norms)

    return np.hstack(blocks)
