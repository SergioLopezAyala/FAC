import argparse
import multiprocessing
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from tta_wbf import VIEW_SETS

KIT_DIR = Path("Entrega_Context/kit_participantes")
GT_DIR = Path("Entrega_Context/spacelab_dataset/labels/test")
MAX_DETECTIONS = 300
SKIP_SCORE = 0.001
CACHES: list[dict] = []


def fuse_image(
    model_entries: list, views: set, weights: list[float], iou: float, conf_type: str
) -> np.ndarray:
    boxes_list, scores_list, labels_list, list_weights = [], [], [], []
    for entries, weight in zip(model_entries, weights, strict=True):
        for view, boxes, scores, labels in entries:
            if view in views:
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)
                list_weights.append(weight)
    if sum(len(scores) for scores in scores_list) == 0:
        return np.zeros((0, 6))
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=list_weights,
        iou_thr=iou,
        skip_box_thr=SKIP_SCORE,
        conf_type=conf_type,
    )
    order = np.argsort(-scores, kind="mergesort")[:MAX_DETECTIONS]
    boxes, scores, labels = boxes[order], scores[order], labels[order]
    width, height = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (width > 0) & (height > 0)
    return np.column_stack(
        [
            labels[keep],
            (boxes[keep, 0] + boxes[keep, 2]) / 2,
            (boxes[keep, 1] + boxes[keep, 3]) / 2,
            width[keep],
            height[keep],
            scores[keep],
        ]
    )


def fuse_chunk(task: tuple) -> dict:
    image_ids, views, weights, iou, conf_type = task
    return {
        image_id: fuse_image([cache[image_id] for cache in CACHES], views, weights, iou, conf_type)
        for image_id in image_ids
    }


def fuse_all(pool, image_ids: list[str], config: tuple) -> dict:
    views, weights, iou, conf_type = config
    chunks = [image_ids[i : i + 25] for i in range(0, len(image_ids), 25)]
    fused = {}
    for part in pool.map(fuse_chunk, [(chunk, views, weights, iou, conf_type) for chunk in chunks]):
        fused.update(part)
    return fused


def score(evaluar, gt: dict, preds: dict) -> tuple[float, float, float, float]:
    prepared = evaluar.preparar(gt, evaluar.Envio(preds=preds))
    threshold, f1 = evaluar.barrer_umbral(prepared)
    total = evaluar.metricas(prepared, threshold)
    return total["score"], total["map50_95"], f1, threshold


def parse_config(text: str) -> tuple:
    views_name, weights_text, iou_text, conf_type = text.split(":")
    weights = [float(w) for w in weights_text.split(",")]
    return set(VIEW_SETS[views_name]), weights, float(iou_text), conf_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse cached multi-view predictions and score fusion settings."
    )
    parser.add_argument("--test-caches", nargs="+", type=Path, required=True)
    parser.add_argument("--oculto-caches", nargs="+", type=Path, default=None)
    parser.add_argument("--configs", nargs="+", required=True, help="views:w1,w2,w3:iou:conf_type")
    parser.add_argument(
        "--write", type=Path, default=None, help="write test and oculto folders for the first config"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--kit", type=Path, default=KIT_DIR)
    return parser.parse_args()


def load(paths: list[Path]) -> list[dict]:
    caches = []
    for path in paths:
        with open(path, "rb") as handle:
            caches.append(pickle.load(handle))
    return caches


def write_folder(formato, preds: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for image_id, rows in preds.items():
        detections = [
            (int(c), xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2, float(s)) for c, xc, yc, w, h, s in rows
        ]
        formato.escribir_txt(str(folder / f"{image_id}.txt"), detections, 1.0, 1.0)


def main() -> int:
    global CACHES
    args = parse_args()
    sys.path.insert(0, str(args.kit.resolve()))
    import evaluar
    import formato_envio

    gt = evaluar.leer_gt(str(GT_DIR))
    CACHES = load(args.test_caches)
    image_ids = sorted(CACHES[0])
    context = multiprocessing.get_context("fork")
    with context.Pool(args.workers) as pool:
        for text in args.configs:
            start = time.time()
            fused = fuse_all(pool, image_ids, parse_config(text))
            total, map5095, f1, threshold = score(evaluar, gt, fused)
            print(
                f"{text:<40} score {total:.4f} mAP50-95 {map5095:.4f} F1 {f1:.4f} thr {threshold:.2f} "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )
            if args.write and text == args.configs[0]:
                write_folder(formato_envio, fused, args.write / "test")
    if args.write and args.oculto_caches:
        CACHES = load(args.oculto_caches)
        oculto_ids = sorted(CACHES[0])
        with context.Pool(args.workers) as pool:
            fused = fuse_all(pool, oculto_ids, parse_config(args.configs[0]))
        write_folder(formato_envio, fused, args.write / "oculto")
        print(f"WROTE {args.write}/test and {args.write}/oculto ({len(oculto_ids)} hidden images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
