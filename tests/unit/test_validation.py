import pytest

from tda_selector.models import RunConfig
from tda_selector.validation import resolve_stop_condition, validate_config


def test_validate_requires_stop_criteria_for_fps() -> None:
    cfg = RunConfig(structures="a.xyz", n_select=None, min_distance=None)
    with pytest.raises(Exception):
        validate_config(cfg)


def test_validate_kmeans_requires_n_select() -> None:
    cfg = RunConfig(structures="a.xyz", selection_method="kmeans", n_select=None)
    with pytest.raises(Exception):
        validate_config(cfg)


def test_resolve_stop_condition() -> None:
    assert resolve_stop_condition("fps", 10, None) == "n_select"
    assert resolve_stop_condition("fps", None, 0.1) == "min_distance"
    assert resolve_stop_condition("fps", 10, 0.1) == "both"
    assert resolve_stop_condition("kmeans", 10, None) == "kmeans_complete"
