import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tta_wbf import IMAGE_SUFFIXES, VIEW_SETS, apply_view, invert_boxes

SHAPE_GROUPS = (("id", "hflip", "vflip", "rot180"), ("rot90", "rot270", "transpose", "antitranspose"))
MAX_DETECTIONS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict every TTA view once and cache the raw boxes.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--views", choices=sorted(VIEW_SETS), default="d8")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    wanted = VIEW_SETS[args.views]
    groups = [[view for view in group if view in wanted] for group in SHAPE_GROUPS]
    groups = [group for group in groups if group]
    images = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[: args.limit]
    cache = {}
    start = time.time()
    for index, path in enumerate(images, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read {path}")
        entries = []
        for group in groups:
            results = model.predict(
                source=[apply_view(image, view) for view in group],
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=MAX_DETECTIONS,
                device=args.device,
                verbose=False,
            )
            for view, result in zip(group, results, strict=True):
                found = result.boxes
                boxes = invert_boxes(found.xyxyn.cpu().numpy().astype(np.float64), view).astype(np.float32)
                entries.append(
                    (
                        view,
                        boxes,
                        found.conf.cpu().numpy().astype(np.float32),
                        found.cls.cpu().numpy().astype(np.int16),
                    )
                )
        cache[path.stem] = entries
        if index % 200 == 0:
            print(f"{index}/{len(images)} images, {time.time() - start:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump(cache, handle)
    print(
        f"DONE {len(images)} images x {sum(len(g) for g in groups)} views -> {args.out} "
        f"in {time.time() - start:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
