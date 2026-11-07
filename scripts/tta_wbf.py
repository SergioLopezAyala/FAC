import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

KIT_DIR = Path("Entrega_Context/kit_participantes")
FLIPS = [(False, False), (True, False), (False, True), (True, True)]
FLIP_SETS = {"none": FLIPS[:1], "h": FLIPS[:2], "all": FLIPS}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_DETECTIONS = 300


def flip_image(image: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
    if horizontal:
        image = image[:, ::-1]
    if vertical:
        image = image[::-1]
    return np.ascontiguousarray(image)


def unflip_boxes(boxes: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
    out = boxes.copy()
    if horizontal:
        out[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
    if vertical:
        out[:, [1, 3]] = 1.0 - boxes[:, [3, 1]]
    return out


def predict_variants(models, image: np.ndarray, flips, args) -> tuple[list, list, list]:
    boxes_list, scores_list, labels_list = [], [], []
    variants = [flip_image(image, horizontal, vertical) for horizontal, vertical in flips]
    for model in models:
        results = model.predict(
            source=variants,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=MAX_DETECTIONS,
            device=args.device,
            verbose=False,
        )
        for (horizontal, vertical), result in zip(flips, results, strict=True):
            found = result.boxes
            boxes = unflip_boxes(found.xyxyn.cpu().numpy(), horizontal, vertical)
            boxes_list.append(np.clip(boxes, 0.0, 1.0))
            scores_list.append(found.conf.cpu().numpy())
            labels_list.append(found.cls.cpu().numpy())
    return boxes_list, scores_list, labels_list


def fuse(boxes_list: list, scores_list: list, labels_list: list, args) -> list[tuple]:
    if sum(len(scores) for scores in scores_list) == 0:
        return []
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=args.wbf_iou, skip_box_thr=args.conf, conf_type="avg"
    )
    order = np.argsort(-scores, kind="mergesort")[:MAX_DETECTIONS]
    return [(int(labels[i]), *boxes[i].tolist(), float(scores[i])) for i in order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flip TTA and multi-model fusion with Weighted Boxes Fusion."
    )
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--imgsz", type=int, nargs="+", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--flips", choices=sorted(FLIP_SETS), default="all")
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
    flips = FLIP_SETS[args.flips]
    images = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(images, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read {path}")
        boxes_list, scores_list, labels_list = [], [], []
        for model, size in zip(models, sizes, strict=True):
            args.imgsz = size
            found = predict_variants([model], image, flips, args)
            boxes_list += found[0]
            scores_list += found[1]
            labels_list += found[2]
        detections = fuse(boxes_list, scores_list, labels_list, args)
        formato_envio.escribir_txt(str(args.out / f"{path.stem}.txt"), detections, 1.0, 1.0)
        if index % 200 == 0:
            print(f"{index}/{len(images)} images", flush=True)
    print(f"DONE {len(images)} images, {len(models)} models x {len(flips)} flips -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
