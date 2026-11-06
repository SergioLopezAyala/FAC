import argparse
import importlib.metadata
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
import sahi
import shapely
import torch
import ultralytics
from rasterio.transform import from_origin
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

DETECT_WEIGHTS = Path("weights/yolo11n.pt")
OBB_WEIGHTS = Path("weights/yolo11n-obb.pt")
SYNTHETIC_IMAGE_SIZE = 2048
SLICE_SIZE = 640
SLICE_OVERLAP = 0.2
DEFAULT_CONFIDENCE = 0.25
LOW_CONFIDENCE = 0.001
OBB_POLYGON_COORDINATE_COUNT = 8
GEO_CRS = "EPSG:32630"
GEO_ORIGIN = (500000.0, 4000000.0)
GEO_PIXEL_SIZE = 10.0


def pixel_to_map(transform, x: float, y: float) -> tuple[float, float]:
    return transform @ (x, y)


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_versions(device: str) -> None:
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "sahi": sahi.__version__,
        "rasterio": rasterio.__version__,
        "cv2": cv2.__version__,
        "shapely": shapely.__version__,
        "numpy": np.__version__,
        "device": device,
    }
    for name, version in versions.items():
        print(f"{name:<12}{version}")


def require_no_headless_opencv() -> None:
    try:
        version = importlib.metadata.version("opencv-python-headless")
    except importlib.metadata.PackageNotFoundError:
        return
    raise RuntimeError(f"opencv-python-headless {version} is installed and conflicts with opencv-python")


def make_synthetic_sea_image(size: int = SYNTHETIC_IMAGE_SIZE, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 6.0, (size, size, 3))
    image = np.clip(np.array([18.0, 52.0, 88.0]) + noise, 0, 255).astype(np.uint8)
    for _ in range(40):
        center = tuple(int(value) for value in rng.integers(0, size, 2))
        half_length = int(rng.integers(8, 24))
        half_width = max(2, half_length // 4)
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(image, center, (half_length, half_width), angle, 0, 360, (235, 235, 225), -1)
    return image


def load_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_model(weights: Path, device: str):
    weights.parent.mkdir(parents=True, exist_ok=True)
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=DEFAULT_CONFIDENCE,
        device=device,
    )


def sliced_prediction(image: np.ndarray, model, confidence: float | None = None):
    return get_sliced_prediction(
        image,
        model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=SLICE_OVERLAP,
        overlap_width_ratio=SLICE_OVERLAP,
        verbose=0,
        confidence_threshold=confidence,
    )


def require_obb_polygons(predictions: list) -> None:
    for prediction in predictions:
        polygon = prediction.mask.segmentation[0]
        if len(polygon) != OBB_POLYGON_COORDINATE_COUNT:
            raise RuntimeError(f"Expected an 8 coordinate OBB polygon, got {len(polygon)} values")


def require_detections(predictions: list, description: str) -> None:
    if not predictions:
        raise RuntimeError(f"No detections from {description} at confidence {LOW_CONFIDENCE}")


def check_inference(name: str, weights: Path, image: np.ndarray, device: str):
    model = load_model(weights, device)
    full_image = get_prediction(image, model).object_prediction_list
    sliced = sliced_prediction(image, model).object_prediction_list
    print(f"{name}: get_prediction={len(full_image)} get_sliced_prediction={len(sliced)}")
    low_confidence_full_image = get_prediction(
        image, model, confidence_threshold=LOW_CONFIDENCE
    ).object_prediction_list
    require_detections(low_confidence_full_image, f"{name} get_prediction")
    require_detections(
        sliced_prediction(image, model, LOW_CONFIDENCE).object_prediction_list,
        f"{name} get_sliced_prediction",
    )
    return model


def write_geotiff(image: np.ndarray, path: Path) -> None:
    height, width, band_count = image.shape
    transform = from_origin(*GEO_ORIGIN, GEO_PIXEL_SIZE, GEO_PIXEL_SIZE)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=band_count,
        dtype=image.dtype,
        crs=GEO_CRS,
        transform=transform,
    ) as dataset:
        dataset.write(np.moveaxis(image, -1, 0))


def check_geotiff_round_trip(image: np.ndarray, model, directory: Path) -> tuple[float, float]:
    path = directory / "smoke.tif"
    write_geotiff(image, path)
    with rasterio.open(path) as dataset:
        array = np.moveaxis(dataset.read(), 0, -1)
        transform = dataset.transform
        bounds = dataset.bounds
        crs = dataset.crs
    if crs != rasterio.CRS.from_string(GEO_CRS):
        raise RuntimeError(f"GeoTIFF CRS changed on read: {crs}")
    if not np.array_equal(array, image):
        raise RuntimeError("GeoTIFF pixels changed on read")

    predictions = sliced_prediction(array, model, LOW_CONFIDENCE).object_prediction_list
    require_detections(predictions, "the GeoTIFF array")
    require_obb_polygons(predictions)

    best = max(predictions, key=lambda prediction: prediction.score.value)
    min_x, min_y, max_x, max_y = best.bbox.to_xyxy()
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    map_x, map_y = pixel_to_map(transform, center_x, center_y)
    if not (bounds.left <= map_x <= bounds.right and bounds.bottom <= map_y <= bounds.top):
        raise RuntimeError(f"Map coordinate ({map_x}, {map_y}) is outside the raster bounds {bounds}")
    print(
        f"geotiff: crs={crs} detections={len(predictions)} best={best.category.name} "
        f"score={best.score.value:.4f} pixel=({center_x:.1f}, {center_y:.1f}) map=({map_x:.1f}, {map_y:.1f})"
    )
    return map_x, map_y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the SAHI, Ultralytics and rasterio stack end to end."
    )
    parser.add_argument("--image", type=Path, help="Real image to use instead of the synthetic sea image.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = select_device()
    print_versions(device)
    require_no_headless_opencv()

    image = load_image(args.image) if args.image else make_synthetic_sea_image()
    print(f"image: shape={image.shape} source={args.image or 'synthetic'}")

    check_inference("detect yolo11n", DETECT_WEIGHTS, image, device)
    obb_model = check_inference("obb yolo11n-obb", OBB_WEIGHTS, image, device)
    if "ship" not in obb_model.category_names:
        raise RuntimeError(f"OBB model has no ship class: {obb_model.category_names}")

    with tempfile.TemporaryDirectory() as directory:
        check_geotiff_round_trip(image, obb_model, Path(directory))

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
