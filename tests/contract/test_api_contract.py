from tda_selector import farthest_point_sampling
import numpy as np


def test_api_surface_basic_selection() -> None:
    features = np.array([[0.0], [1.0], [2.0]])
    selected, _ = farthest_point_sampling(features, n_select=2, start="0")
    assert selected[0] == 0
    assert len(selected) == 2
