import numpy as np
import pytest
from numpy.testing import assert_allclose
from tta_wbf import VIEW_SETS, VIEWS, apply_view, invert_boxes


def test_horizontal_flip_inverse_mirrors_x_coordinates():
    boxes = np.array([[0.1, 0.2, 0.3, 0.5]])
    assert_allclose(invert_boxes(boxes, "hflip"), [[0.7, 0.2, 0.9, 0.5]])


def test_vertical_flip_inverse_mirrors_y_coordinates():
    boxes = np.array([[0.1, 0.2, 0.3, 0.5]])
    assert_allclose(invert_boxes(boxes, "vflip"), [[0.1, 0.5, 0.3, 0.8]])


@pytest.mark.parametrize("view", VIEWS)
def test_box_found_on_transformed_image_maps_back_to_original(view):
    image = np.zeros((40, 100, 3), dtype=np.uint8)
    image[10:20, 5:25] = 255
    transformed = apply_view(image, view)
    ys, xs = np.nonzero(transformed[:, :, 0])
    height, width = transformed.shape[:2]
    found = np.array([[xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height]])
    assert_allclose(invert_boxes(found, view), [[0.05, 0.25, 0.25, 0.5]])


@pytest.mark.parametrize("view", VIEWS)
def test_inverted_boxes_keep_valid_corner_order(view):
    out = invert_boxes(np.array([[0.05, 0.4, 0.25, 0.9]]), view)
    assert np.all(out[:, 2] > out[:, 0])
    assert np.all(out[:, 3] > out[:, 1])


def test_view_sets_only_use_known_views():
    assert all(view in VIEWS for views in VIEW_SETS.values() for view in views)
    assert VIEW_SETS["flips4"] == ["id", "hflip", "vflip", "rot180"]
    assert len(VIEW_SETS["d8"]) == 8
