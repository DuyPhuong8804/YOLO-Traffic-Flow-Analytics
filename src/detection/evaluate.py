"""
Danh gia mo hinh da fine-tune tren tap validation/test: mAP@0.5, Precision, Recall.
Muc tieu de bai: mAP@0.5 >= 0.7.

Chay:
    python -m src.detection.evaluate --weights models/vehicle_yolo26/weights/best.pt
"""

import argparse

from ultralytics import YOLO

from src.common.config_loader import load_yaml


def evaluate(weights: str, data_config: str = "train_config.yaml", split: str = "test") -> None:
    cfg = load_yaml(data_config)
    model = YOLO(weights)

    metrics = model.val(data=cfg["data"], split=split)

    print("\n===== KET QUA DANH GIA =====")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")

    target_map = 0.7
    status = "DAT" if metrics.box.map50 >= target_map else "CHUA DAT"
    print(f"\nMuc tieu mAP@0.5 >= {target_map}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danh gia model detect car/truck/motorbike")
    parser.add_argument("--weights", required=True, help="Duong dan file .pt da train")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    evaluate(args.weights, split=args.split)
