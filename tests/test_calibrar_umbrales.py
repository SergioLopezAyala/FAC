from types import SimpleNamespace

import numpy as np
import pytest
from calibrar_umbrales import class_f1_curve, plateau_middle, rescale


def test_class_f1_curve_matches_official_formula():
    per_class = SimpleNamespace(
        conf=np.array([0.9, 0.8, 0.7]),
        tp=np.array([[True, False, True]]),
        n_gt_img={"a": 3, "b": 1},
    )
    f1 = class_f1_curve(per_class, np.array([0.95, 0.85, 0.75, 0.65]))
    assert f1 == pytest.approx([0.0, 2 / 5, 2 / 6, 4 / 7])


def test_class_threshold_maps_to_global_threshold():
    assert rescale(np.array([0.3]), 0.3, 0.5)[0] == pytest.approx(0.5)


def test_endpoints_are_preserved():
    assert rescale(np.array([0.0, 1.0]), 0.3, 0.5) == pytest.approx([0.0, 1.0])


def test_rescale_is_strictly_increasing_on_both_sides_of_threshold():
    scores = np.linspace(0.0, 1.0, 101)
    for class_threshold, target in [(0.2, 0.5), (0.7, 0.5), (0.5, 0.5), (0.05, 0.9)]:
        assert np.all(np.diff(rescale(scores, class_threshold, target)) > 0)


def test_selection_after_rescale_matches_selection_at_class_threshold():
    scores = np.array([0.1, 0.29, 0.3, 0.31, 0.8])
    assert np.array_equal(rescale(scores, 0.3, 0.5) >= 0.5, scores >= 0.3)


def test_plateau_middle_picks_center_of_flat_maximum():
    thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    f1 = np.array([0.50, 0.70, 0.701, 0.702, 0.60, 0.40])
    assert plateau_middle(thresholds, f1, tolerance=0.005) == pytest.approx(0.3)


def test_plateau_middle_with_single_peak_returns_peak():
    thresholds = np.array([0.1, 0.2, 0.3])
    f1 = np.array([0.2, 0.9, 0.3])
    assert plateau_middle(thresholds, f1, tolerance=0.005) == pytest.approx(0.2)
