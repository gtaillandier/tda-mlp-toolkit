from pathlib import Path
from typing import Optional

import ase.io
import numpy as np
from ase import Atoms


def auto_output_dir(
    per_species: bool,
    supercell: tuple[int, int, int],
    max_edge_length: float,
    bandwidth: float,
    resolution: tuple[int, int],
    selection_method: str,
    n_select: Optional[int],
    min_distance: Optional[float],
) -> str:
    per_str = f"per{'T' if per_species else 'F'}"
    sc = f"sc{supercell[0]}x{supercell[1]}x{supercell[2]}"
    mel = f"mel{max_edge_length}"
    bw = f"bw{bandwidth}"
    res = f"res{resolution[0]}x{resolution[1]}"
    if selection_method == "kmeans":
        conv = f"kmeans_n{n_select}"
    elif n_select is not None and min_distance is not None:
        conv = f"n{n_select}_d{min_distance}"
    elif n_select is not None:
        conv = f"n{n_select}"
    else:
        conv = f"d{min_distance}"
    return f"tda_{per_str}_{sc}_{mel}_{bw}_{res}_{conv}"


def save_results(
    selected: list[int],
    dist_history: list[float],
    output_dir: Path,
    structures_path: str,
    structures: list[Atoms],
    stopping_condition: str,
    extra_summary: Optional[list[str]] = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    idx_path = output_dir / "selected_indices.txt"
    np.savetxt(idx_path, selected, fmt="%d")

    if dist_history:
        conv = output_dir / "convergence.txt"
        np.savetxt(
            conv,
            np.column_stack([np.arange(1, len(dist_history) + 1), dist_history]),
            fmt=["%-6d", "%.8f"],
            header="step  coverage_radius",
            comments="",
        )

    sel_path = output_dir / "selected_structures.xyz"
    ase.io.write(sel_path, [structures[i] for i in selected])

    summary_lines = [
        f"Source:            {structures_path}",
        f"Total structures:  {len(structures)}",
        f"Selected:          {len(selected)}",
        f"Fraction selected: {len(selected)/len(structures):.3%}",
        f"Stopping condition:{stopping_condition}",
    ]
    if dist_history:
        summary_lines += [
            f"Initial radius:    {dist_history[0]:.6f}",
            f"Final radius:      {dist_history[-1]:.6f}",
        ]
    if extra_summary:
        summary_lines += extra_summary

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "indices": idx_path,
        "selected_structures": sel_path,
        "summary": summary_path,
    }
