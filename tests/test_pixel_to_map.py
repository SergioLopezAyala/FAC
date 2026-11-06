import pytest
from rasterio.transform import Affine, from_origin
from smoke_test_sahi import pixel_to_map


def test_north_up_transform_scales_and_offsets_pixel_coordinates():
    transform = from_origin(500000, 4000000, 10, 10)

    assert pixel_to_map(transform, 100, 50) == pytest.approx((501000, 3999500))


def test_rotated_transform_applies_shear_terms():
    transform = Affine(10, 2, 500000, 3, -10, 4000000)

    assert pixel_to_map(transform, 100, 50) == pytest.approx((501100, 3999800))


def test_origin_pixel_corner_maps_to_transform_origin_without_half_pixel_shift():
    transform = from_origin(500000, 4000000, 10, 10)

    assert pixel_to_map(transform, 0, 0) == pytest.approx((500000, 4000000))
