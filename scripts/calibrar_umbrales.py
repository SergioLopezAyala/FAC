import argparse
import json
import sys
from pathlib import Path

import numpy as np

KIT_DIR = Path("Entrega_Context/kit_participantes")
THRESHOLD_GRID = np.round(np.arange(0.01, 0.96, 0.01), 2)
PLATEAU_TOLERANCE = 0.005
MAP_TOLERANCE = 0.0005
MAX_DETECTIONS = 300


def rescale(scores: np.ndarray, class_threshold: float, target: float) -> np.ndarray:
    below = scores * target / class_threshold
    above = target + (1.0 - target) * (scores - class_threshold) / (1.0 - class_threshold)
    return np.where(scores < class_threshold, below, above)


def plateau_middle(thresholds: np.ndarray, f1: np.ndarray, tolerance: float) -> float:
    near_best = np.nonzero(f1 >= f1.max() - tolerance)[0]
    return float(thresholds[near_best[len(near_best) // 2]])


def class_f1_curve(per_class, thresholds: np.ndarray) -> np.ndarray:
    n_gt = sum(per_class.n_gt_img.values())
    order = np.argsort(-per_class.conf, kind="mergesort")
    conf, tp = per_class.conf[order], per_class.tp[0, order]
    tp_cumulative = np.concatenate([[0], np.cumsum(tp)])
    n_selected = np.searchsorted(-conf, -thresholds, side="right")
    n_tp = tp_cumulative[n_selected]
    return 2 * n_tp / np.maximum(n_selected + n_gt, 1)


def class_thresholds(prepared, target: float) -> list[float]:
    thresholds = []
    for per_class in prepared:
        if sum(per_class.n_gt_img.values()) == 0:
            thresholds.append(target)
            continue
        f1 = class_f1_curve(per_class, THRESHOLD_GRID)
        thresholds.append(plateau_middle(THRESHOLD_GRID, f1, PLATEAU_TOLERANCE))
    return thresholds


def calibrate_folder(preds: dict, out_dir: Path, thresholds: list[float], target: float, formato) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_id, boxes in preds.items():
        kept = boxes[np.argsort(-boxes[:, 5], kind="mergesort")[:MAX_DETECTIONS]]
        detections = []
        for cls, xc, yc, w, h, score in kept:
            new_score = float(rescale(np.array([score]), thresholds[int(cls)], target)[0])
            detections.append((int(cls), xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2, new_score))
        formato.escribir_txt(str(out_dir / f"{image_id}.txt"), detections, 1.0, 1.0)


def summarize(label: str, result: dict) -> None:
    total = result["total"]
    print(
        f"{label}: score {total['score']:.4f} mAP50-95 {total['map50_95']:.4f} "
        f"mAP50 {total['map50']:.4f} F1 {total['f1_macro']:.4f} threshold {result['umbral']:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-class confidence calibration behind one global threshold."
    )
    parser.add_argument("--test-preds", type=Path, required=True)
    parser.add_argument("--oculto-preds", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=Path("Entrega_Context/spacelab_dataset/labels/test"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.50)
    parser.add_argument("--kit", type=Path, default=KIT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.kit.resolve()))
    import evaluar
    import formato_envio

    if round(args.target, 2) != args.target or not 0 < args.target < 1:
        raise ValueError("--target must be between 0 and 1 with at most 2 decimals")

    gt = evaluar.leer_gt(str(args.gt))
    meta = evaluar.leer_meta(None, gt)
    original = evaluar.leer_envio(str(args.test_preds))
    prepared = evaluar.preparar(gt, original)
    best_single, _ = evaluar.barrer_umbral(prepared)
    baseline = evaluar.evaluar(gt, original, meta, umbral=best_single)

    thresholds = class_thresholds(prepared, args.target)
    calibrate_folder(original.preds, args.out / "test", thresholds, args.target, formato_envio)
    calibrated = evaluar.evaluar(gt, evaluar.leer_envio(str(args.out / "test")), meta, umbral=args.target)

    summarize("single threshold", baseline)
    summarize("per-class calibrated", calibrated)
    print("class thresholds:", dict(zip(evaluar.CLASES, thresholds, strict=True)))
    map_change = calibrated["total"]["map50_95"] - baseline["total"]["map50_95"]
    f1_change = calibrated["total"]["f1_macro"] - baseline["total"]["f1_macro"]
    print(f"change: mAP50-95 {map_change:+.5f} F1 {f1_change:+.5f}")

    report = {
        "target": args.target,
        "best_single_threshold": best_single,
        "class_thresholds": dict(zip(evaluar.CLASES, thresholds, strict=True)),
        "single": baseline["total"],
        "calibrated": calibrated["total"],
    }
    (args.out / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if map_change < -MAP_TOLERANCE or f1_change <= 0:
        print("REJECTED: calibration lowered mAP or did not raise F1; keep the single-threshold submission")
        return 1

    calibrate_folder(
        evaluar.leer_envio(str(args.oculto_preds)).preds,
        args.out / "oculto",
        thresholds,
        args.target,
        formato_envio,
    )
    zip_path = args.out / "envio_calibrado.zip"
    count = formato_envio.crear_zip(str(args.out / "oculto"), str(zip_path), umbral=args.target)
    print(f"ACCEPTED: {zip_path} with {count} files, umbral {args.target:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
