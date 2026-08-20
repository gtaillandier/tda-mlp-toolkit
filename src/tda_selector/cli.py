import argparse
import sys

from .api import select_structures
from .errors import SelectorError
from .models import RunConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TDA-based diversity selector")
    p.add_argument("structures")
    p.add_argument("--index", default=":")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--features-cache", default=None)

    p.add_argument("--max-edge-length", type=float, default=6.0)
    p.add_argument("--homology-dims", nargs="+", type=int, default=[0, 1])
    p.add_argument("--bandwidth", type=float, default=0.5)
    p.add_argument("--resolution", nargs=2, type=int, default=[20, 20], metavar=("W", "H"))
    p.add_argument("--no-per-species", dest="per_species", action="store_false")
    p.add_argument("--species", nargs="+", default=None)
    p.add_argument("--max-atoms-per-species", type=int, default=None)
    p.add_argument("--supercell", nargs=3, type=int, default=[1, 1, 1], metavar=("NA", "NB", "NC"))
    p.add_argument("--n-jobs", type=int, default=-1)

    p.add_argument("--selection-method", choices=["fps", "kmeans"], default="fps")
    p.add_argument("--n-select", type=int, default=None)
    p.add_argument("--min-distance", type=float, default=None)
    p.add_argument("--start", default="centroid")
    p.add_argument("--no-fill", action="store_true", default=False)

    p.add_argument("--plot", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = RunConfig(
        structures=args.structures,
        index=args.index,
        output_dir=args.output_dir,
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
        selection_method=args.selection_method,
        n_select=args.n_select,
        min_distance=args.min_distance,
        start=args.start,
        no_fill=args.no_fill,
        plot=args.plot,
        seed=args.seed,
    )

    try:
        result = select_structures(cfg)
    except SelectorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    print(f"Selected {len(result.selected_indices)} structures")
    print(f"Saved outputs to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
