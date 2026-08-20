import hashlib
import json
from .models import RunConfig


def seed_everything(seed: int) -> None:
    import numpy as np

    np.random.seed(seed)


def cache_fingerprint(cfg: RunConfig, n_structures: int) -> str:
    key = {
        "n": n_structures,
        "max_edge_length": cfg.max_edge_length,
        "homology_dims": cfg.homology_dims,
        "bandwidth": cfg.bandwidth,
        "resolution": cfg.resolution,
        "per_species": cfg.per_species,
        "species": cfg.species,
        "supercell": cfg.supercell,
        "max_atoms_per_species": cfg.max_atoms_per_species,
    }
    blob = json.dumps(key, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
