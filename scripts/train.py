"""Fine-tune YOLO26 tren bo du lieu Vietnamese vehicle (3 lop: car, truck, motorbike)."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from ultralytics import YOLO  # noqa: E402

from traffic.config import load_config  # noqa: E402


def main():
    cfg = load_config()["train"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--batch", type=int, default=cfg["batch"])
    ap.add_argument("--imgsz", type=int, default=cfg["imgsz"])
    ap.add_argument("--model", default=cfg["base_model"])
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=cfg["data"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=cfg["patience"],
        project=str(ROOT / cfg["project"]),
        name=cfg["name"],
        exist_ok=True,
        device=args.device,
        workers=2,
        hsv_v=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        plots=True,
    )


if __name__ == "__main__":
    main()
