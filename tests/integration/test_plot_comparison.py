from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from tda_selector.compare_cli import main  # noqa: E402


def _write_traj(path: Path) -> None:
    frames = [Atoms("H2", positions=[[0, 0, 0], [0, 0, d]]) for d in np.linspace(0.7, 1.5, 8)]
    ase.io.write(path, frames)


def test_compare_plot_runs_both_methods(tmp_path: Path) -> None:
    traj = tmp_path / "small.xyz"
    _write_traj(traj)
    out = tmp_path / "cmp.png"

    rc = main([str(traj), "--n-select", "3", "--homology-dims", "0", "--resolution", "5", "5",
               "--n-jobs", "1", "--output", str(out), "--save-indices", str(tmp_path / "idx")])

    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    assert len(np.loadtxt(tmp_path / "idx" / "fps_indices.txt")) == 3
    assert len(np.loadtxt(tmp_path / "idx" / "kmeans_indices.txt")) == 3


def test_compare_plot_uses_existing_indices(tmp_path: Path) -> None:
    traj = tmp_path / "small.xyz"
    _write_traj(traj)
    fps = tmp_path / "fps.txt"
    km = tmp_path / "km.txt"
    np.savetxt(fps, [0, 7], fmt="%d")
    np.savetxt(km, [2, 5], fmt="%d")
    out = tmp_path / "cmp.png"

    rc = main([str(traj), "--homology-dims", "0", "--resolution", "5", "5", "--n-jobs", "1",
               "--fps-indices", str(fps), "--kmeans-indices", str(km), "--output", str(out)])

    assert rc == 0
    assert out.exists()
