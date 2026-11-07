import numpy as np
from numpy.testing import assert_allclose
from tta_wbf import FLIPS, flip_image, unflip_boxes


def test_horizontal_unflip_mirrors_x_coordinates():
    boxes = np.array([[0.1, 0.2, 0.3, 0.5]])
    assert_allclose(unflip_boxes(boxes, horizontal=True, vertical=False), [[0.7, 0.2, 0.9, 0.5]])


def test_vertical_unflip_mirrors_y_coordinates():
    boxes = np.array([[0.1, 0.2, 0.3, 0.5]])
    assert_allclose(unflip_boxes(boxes, horizontal=False, vertical=True), [[0.1, 0.5, 0.3, 0.8]])


def test_unflip_keeps_corner_order_valid():
    boxes = np.array([[0.05, 0.4, 0.25, 0.9]])
    out = unflip_boxes(boxes, horizontal=True, vertical=True)
    assert np.all(out[:, 2] > out[:, 0])
    assert np.all(out[:, 3] > out[:, 1])


def test_box_on_flipped_image_maps_back_to_original_position():
    image = np.zeros((40, 100, 3), dtype=np.uint8)
    image[10:20, 5:25] = 255
    for horizontal, vertical in FLIPS:
        flipped = flip_image(image, horizontal, vertical)
        ys, xs = np.nonzero(flipped[:, :, 0])
        found = np.array([[xs.min() / 100, ys.min() / 40, (xs.max() + 1) / 100, (ys.max() + 1) / 40]])
        assert_allclose(unflip_boxes(found, horizontal, vertical), [[0.05, 0.25, 0.25, 0.5]])
