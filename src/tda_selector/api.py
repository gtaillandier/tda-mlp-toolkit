from pathlib import Path

import numpy as np

from .errors import CacheError
from .features import compute_all_diagrams, diagrams_to_features, load_structures
from .io import auto_output_dir, save_results
from .models import RunConfig, RunResult
from .reproducibility import cache_fingerprint, seed_everything
from .selection import farthest_point_sampling, kmeans_selection
from .validation import resolve_stop_condition, validate_config


def _load_or_build_features(structures, cfg: RunConfig) -> np.ndarray:
    cache = Path(cfg.features_cache) if cfg.features_cache else None
    fp_path = cache.with_suffix(cache.suffix + ".sha256") if cache else None

    if cache and cache.exists():
        features = np.load(cache)
        expected = cache_fingerprint(cfg, len(structures))
        if fp_path and fp_path.exists() and fp_path.read_text(encoding="utf-8").strip() != expected:
            raise CacheError("Feature cache exists but is incompatible with current settings")
        if features.shape[0] != len(structures):
            raise CacheError("Feature cache row count does not match number of structures")
        return features

    diagrams = compute_all_diagrams(
        structures,
        max_edge_length=cfg.max_edge_length,
        max_dim=max(cfg.homology_dims) + 1,
        per_species=cfg.per_species,
        max_atoms_per_species=cfg.max_atoms_per_species,
        species_filter=cfg.species,
        supercell=cfg.supercell,
        n_jobs=cfg.n_jobs,
    )
    features = diagrams_to_features(
        diagrams,
        homology_dims=cfg.homology_dims,
        bandwidth=cfg.bandwidth,
        resolution=cfg.resolution,
    )

    if cache:
        np.save(cache, features)
        if fp_path:
            fp_path.write_text(cache_fingerprint(cfg, len(structures)), encoding="utf-8")

    return features


def load_features(cfg: RunConfig) -> tuple[list, np.ndarray]:
    structures = load_structures(cfg.structures, index=cfg.index)
    return structures, _load_or_build_features(structures, cfg)


def select_structures(cfg: RunConfig) -> RunResult:
    validate_config(cfg)
    seed_everything(cfg.seed)

    structures = load_structures(cfg.structures, index=cfg.index)
    features = _load_or_build_features(structures, cfg)

    if cfg.selection_method == "kmeans":
        selected, km_metrics = kmeans_selection(features, cfg.n_select or 0, seed=cfg.seed, fill_empty=not cfg.no_fill)
        history = []
        extra = [
            "Selection method:  kmeans",
            f"K-means inertia:   {km_metrics['inertia']:.4f}",
            f"Mean dist/centroid:{km_metrics['mean_dist_to_centroid']:.6f}",
            f"Empty clusters:    {km_metrics['empty_clusters']}",
        ]
    else:
        selected, history = farthest_point_sampling(
            features,
            n_select=cfg.n_select,
            min_distance=cfg.min_distance,
            start=cfg.start,
        )
        extra = None

    out_name = cfg.output_dir or auto_output_dir(
        cfg.per_species,
        cfg.supercell,
        cfg.max_edge_length,
        cfg.bandwidth,
        cfg.resolution,
        cfg.selection_method,
        cfg.n_select,
        cfg.min_distance,
    )
    out_dir = Path(out_name)

    stop = resolve_stop_condition(cfg.selection_method, cfg.n_select, cfg.min_distance)
    paths = save_results(selected, history, out_dir, cfg.structures, structures, stop, extra_summary=extra)

    return RunResult(
        selected_indices=selected,
        output_dir=out_dir,
        summary_path=paths["summary"],
        selected_structures_path=paths["selected_structures"],
        indices_path=paths["indices"],
        stopping_condition=stop,
    )
