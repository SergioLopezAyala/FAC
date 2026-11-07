import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

KIT_DIR = Path("Entrega_Context/kit_participantes")
VIEWS = ("id", "hflip", "vflip", "rot180", "rot90", "rot270", "transpose", "antitranspose")
VIEW_SETS = {
    "none": ["id"],
    "h": ["id", "hflip"],
    "flips4": ["id", "hflip", "vflip", "rot180"],
    "team4": ["id", "hflip", "vflip", "rot90"],
    "d8": list(VIEWS),
}
LEGACY_FLIPS = {"none": "none", "h": "h", "all": "flips4"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_DETECTIONS = 300


def apply_view(image: np.ndarray, view: str) -> np.ndarray:
    transforms = {
        "id": lambda a: a,
        "hflip": lambda a: a[:, ::-1],
        "vflip": lambda a: a[::-1],
        "rot180": lambda a: a[::-1, ::-1],
        "rot90": lambda a: np.rot90(a, 1),
        "rot270": lambda a: np.rot90(a, 3),
        "transpose": lambda a: a.transpose(1, 0, 2),
        "antitranspose": lambda a: np.rot90(a, 2).transpose(1, 0, 2),
    }
    return np.ascontiguousarray(transforms[view](image))


def invert_boxes(boxes: np.ndarray, view: str) -> np.ndarray:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    mappings = {
        "id": (x1, y1, x2, y2),
        "hflip": (1 - x2, y1, 1 - x1, y2),
        "vflip": (x1, 1 - y2, x2, 1 - y1),
        "rot180": (1 - x2, 1 - y2, 1 - x1, 1 - y1),
        "rot90": (1 - y2, x1, 1 - y1, x2),
        "rot270": (y1, 1 - x2, y2, 1 - x1),
        "transpose": (y1, x1, y2, x2),
        "antitranspose": (1 - y2, 1 - x2, 1 - y1, 1 - x1),
    }
    return np.clip(np.stack(mappings[view], axis=1), 0.0, 1.0)


def predict_views(model, image: np.ndarray, views: list[str], size: int, args) -> tuple[list, list, list]:
    boxes_list, scores_list, labels_list = [], [], []
    for view in views:
        result = model.predict(
            source=apply_view(image, view),
            imgsz=size,
            conf=args.conf,
            iou=args.iou,
            max_det=MAX_DETECTIONS,
            device=args.device,
            verbose=False,
        )[0]
        found = result.boxes
        boxes_list.append(invert_boxes(found.xyxyn.cpu().numpy(), view))
        scores_list.append(found.conf.cpu().numpy())
        labels_list.append(found.cls.cpu().numpy())
    return boxes_list, scores_list, labels_list


def fuse(boxes_list: list, scores_list: list, labels_list: list, args) -> list[tuple]:
    if sum(len(scores) for scores in scores_list) == 0:
        return []
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=args.wbf_iou,
        skip_box_thr=args.conf,
        conf_type=args.conf_type,
    )
    order = np.argsort(-scores, kind="mergesort")[:MAX_DETECTIONS]
    return [(int(labels[i]), *boxes[i].tolist(), float(scores[i])) for i in order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TTA views and multi-model fusion with Weighted Boxes Fusion."
    )
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--imgsz", type=int, nargs="+", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", choices=sorted(VIEW_SETS), default=None)
    parser.add_argument("--flips", choices=sorted(LEGACY_FLIPS), default="all")
    parser.add_argument("--conf-type", choices=["avg", "max", "box_and_model_avg"], default="avg")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--wbf-iou", type=float, default=0.55)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--kit", type=Path, default=KIT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.kit.resolve()))
    import formato_envio
    from ultralytics import YOLO

    if len(args.imgsz) not in (1, len(args.weights)):
        raise ValueError("--imgsz takes one value or one per weights file")
    sizes = args.imgsz * len(args.weights) if len(args.imgsz) == 1 else args.imgsz
    models = [YOLO(weights) for weights in args.weights]
    views = VIEW_SETS[args.views or LEGACY_FLIPS[args.flips]]
    images = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(images, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read {path}")
        boxes_list, scores_list, labels_list = [], [], []
        for model, size in zip(models, sizes, strict=True):
            found = predict_views(model, image, views, size, args)
            boxes_list += found[0]
            scores_list += found[1]
            labels_list += found[2]
        detections = fuse(boxes_list, scores_list, labels_list, args)
        formato_envio.escribir_txt(str(args.out / f"{path.stem}.txt"), detections, 1.0, 1.0)
        if index % 200 == 0:
            print(f"{index}/{len(images)} images", flush=True)
    print(
        f"DONE {len(images)} images, {len(models)} models x {len(views)} views "
        f"({','.join(views)}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
