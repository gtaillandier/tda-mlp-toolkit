import numpy as np

from tda_selector.features import diagrams_to_features


def test_diagrams_to_features_shape() -> None:
    d0 = {"H": {0: np.empty((0, 2)), 1: np.empty((0, 2))}}
    d1 = {"H": {0: np.empty((0, 2)), 1: np.empty((0, 2))}}
    feats = diagrams_to_features([d0, d1], homology_dims=(0, 1), bandwidth=0.5, resolution=(5, 5))
    assert feats.shape == (2, 50)
