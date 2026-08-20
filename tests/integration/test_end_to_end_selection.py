from pathlib import Path

import ase.io
from ase import Atoms

from tda_selector.api import select_structures
from tda_selector.models import RunConfig


def test_end_to_end_small_xyz(tmp_path: Path) -> None:
    traj = tmp_path / "small.xyz"
    frames = [
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]),
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.80]]),
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.90]]),
    ]
    ase.io.write(traj, frames)

    cfg = RunConfig(structures=str(traj), n_select=2, output_dir=str(tmp_path / "out"), homology_dims=(0,), resolution=(5, 5))
    res = select_structures(cfg)

    assert len(res.selected_indices) == 2
    assert res.indices_path.exists()
    assert res.summary_path.exists()
    assert res.selected_structures_path.exists()
