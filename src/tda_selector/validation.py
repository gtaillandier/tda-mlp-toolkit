from .errors import ValidationError
from .models import RunConfig


def validate_config(cfg: RunConfig) -> None:
    if cfg.max_edge_length <= 0:
        raise ValidationError("--max-edge-length must be > 0")
    if cfg.bandwidth <= 0:
        raise ValidationError("--bandwidth must be > 0")
    if cfg.resolution[0] <= 0 or cfg.resolution[1] <= 0:
        raise ValidationError("--resolution must be positive")
    if any(s <= 0 for s in cfg.supercell):
        raise ValidationError("--supercell values must be >= 1")
    if cfg.max_atoms_per_species is not None and cfg.max_atoms_per_species <= 0:
        raise ValidationError("--max-atoms-per-species must be > 0")

    if cfg.selection_method == "kmeans":
        if cfg.n_select is None:
            raise ValidationError("--selection-method kmeans requires --n-select")
    else:
        if cfg.n_select is None and cfg.min_distance is None:
            raise ValidationError("Specify at least one of --n-select or --min-distance")
        if cfg.min_distance is not None and cfg.min_distance < 0:
            raise ValidationError("--min-distance must be >= 0")


def resolve_stop_condition(selection_method: str, n_select: int | None, min_distance: float | None) -> str:
    if selection_method == "kmeans":
        return "kmeans_complete"
    if n_select is not None and min_distance is not None:
        return "both"
    if n_select is not None:
        return "n_select"
    return "min_distance"
