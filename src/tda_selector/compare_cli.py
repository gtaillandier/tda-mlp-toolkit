import argparse
import sys
from pathlib import Path

import numpy as np

from .api import load_features
from .cli import add_feature_args, add_input_args
from .errors import SelectorError, ValidationError
from .models import RunConfig
from .plotting import nearest_pick_labels, plot_selection_comparison
from .reproducibility import seed_everything
from .selection import farthest_point_sampling, kmeans_selection


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot the full dataset with FPS and K-means selections overlaid (2D PCA of TDA features)"
    )
    add_input_args(p)
    add_feature_args(p)
    p.add_argument("--n-select", type=int, default=None)
    p.add_argument("--min-distance", type=float, default=None, help="FPS only; K-means then uses the FPS count")
    p.add_argument("--start", default="centroid")
    p.add_argument("--no-fill", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps-indices", default=None, help="Existing selected_indices.txt from an FPS run")
    p.add_argument("--kmeans-indices", default=None, help="Existing selected_indices.txt from a K-means run")
    p.add_argument("--output", default="selection_comparison.png")
    p.add_argument("--cluster-colors", action="store_true",
                   help="Colour structures by K-means cluster instead of plain grey")
    p.add_argument("--save-indices", default=None, help="Directory to write fps_indices.txt and kmeans_indices.txt")
    return p


def _read_indices(path: str, n: int) -> list[int]:
    idx = np.atleast_1d(np.loadtxt(path, dtype=int)).tolist()
    if any(i < 0 or i >= n for i in idx):
        raise ValidationError(f"{path} contains indices outside [0, {n})")
    return idx


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.fps_indices is None and args.n_select is None and args.min_distance is None:
        print("Error: specify --n-select/--min-distance or pass --fps-indices", file=sys.stderr)
        return 2

    cfg = RunConfig(
        structures=args.structures,
        index=args.index,
        features_cache=args.features_cache,
        max_edge_length=args.max_edge_length,
        homology_dims=tuple(args.homology_dims),
        bandwidth=args.bandwidth,
        resolution=tuple(args.resolution),
        per_species=args.per_species,
        species=args.species,
        max_atoms_per_species=args.max_atoms_per_species,
        supercell=tuple(args.supercell),
        n_jobs=args.n_jobs,
        n_select=args.n_select,
        min_distance=args.min_distance,
        start=args.start,
        seed=args.seed,
    )

    try:
        seed_everything(cfg.seed)
        _, features = load_features(cfg)
        n = len(features)

        if args.fps_indices:
            fps = _read_indices(args.fps_indices, n)
        else:
            fps, _ = farthest_point_sampling(features, n_select=args.n_select, min_distance=args.min_distance, start=args.start)

        labels = None
        if args.kmeans_indices:
            km = _read_indices(args.kmeans_indices, n)
        else:
            km, km_info = kmeans_selection(features, args.n_select or len(fps), seed=args.seed, fill_empty=not args.no_fill)
            labels = km_info["labels"]
        if args.cluster_colors and labels is None:
            labels = nearest_pick_labels(features, km)
        elif not args.cluster_colors:
            labels = None

        if args.save_indices:
            out_dir = Path(args.save_indices)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_dir / "fps_indices.txt", fps, fmt="%d")
            np.savetxt(out_dir / "kmeans_indices.txt", km, fmt="%d")

        out = plot_selection_comparison(features, fps, km, args.output, seed=args.seed, cluster_labels=labels)
    except (SelectorError, ImportError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"FPS: {len(fps)}  K-means: {len(km)}  common: {len(set(fps) & set(km))}")
    print(f"Saved plot to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
