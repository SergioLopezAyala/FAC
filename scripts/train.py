import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

DATA = Path("Entrega_Context/spacelab_dataset/data.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time-boxed Ultralytics training with settings for short runs."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extra = {"time": args.hours} if args.hours else {}
    model = YOLO(args.model)
    model.train(
        data=str(DATA.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        fraction=args.fraction,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        patience=100,
        cos_lr=True,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        seed=0,
        flipud=0.5,
        fliplr=0.5,
        plots=True,
        **extra,
    )
    print(f"BEST_WEIGHTS {model.trainer.best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
