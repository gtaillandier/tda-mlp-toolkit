import numpy as np

from tda_selector.selection import farthest_point_sampling


def test_fps_returns_unique_indices() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    selected, _ = farthest_point_sampling(features, n_select=3, start="0")
    assert len(selected) == 3
    assert len(set(selected)) == 3


def test_fps_obeys_min_distance() -> None:
    features = np.array([[0.0], [0.01], [0.02]])
    selected, _ = farthest_point_sampling(features, min_distance=1.0, start="0")
    assert len(selected) == 1


def test_fps_normalizes_negative_start() -> None:
    features = np.array([[0.0], [1.0], [5.0]])
    selected, _ = farthest_point_sampling(features, n_select=1, start="-1")
    assert selected == [2]
