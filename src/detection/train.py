"""
Script fine-tune YOLO26 (Ultralytics) tren dataset car/truck/motorbike.

YOLO26 la mo hinh detect NMS-free (loai bo buoc Non-Max-Suppression hau xu ly
truyen thong nho co che end-to-end dua tren gan nhan one-to-one khi train),
giup giam do tre inference dang ke so voi cac ban YOLO truoc - phu hop yeu cau
real-time >=15 FPS cua he thong.

Chay:
    python -m src.detection.train
    python -m src.detection.train --config configs/train_config.yaml
"""

import argparse

from ultralytics import YOLO

from src.common.config_loader import PROJECT_ROOT, load_yaml


def train(config_path: str = "train_config.yaml") -> None:
    cfg = load_yaml(config_path)

    model = YOLO(cfg["model"])

    # QUAN TRONG: Ultralytics tu dong noi them "<runs_dir>/<task>/" phia truoc neu
    # "project" la duong dan TUONG DOI (xem ultralytics/cfg/__init__.py::get_save_dir),
    # vi du project="models" se ra "runs/detect/models/..." thay vi "models/...".
    # Resolve ve duong dan TUYET DOI de ket qua nam dung noi README/app_config.yaml
    # da mo ta (models/<name>/weights/best.pt).
    project_dir = str(PROJECT_ROOT / cfg["project"])

    # Cac tham so augmentation/training duoc truyen truc tiep vao model.train().
    # Ultralytics tu dong luu checkpoint tot nhat (best.pt) va cuoi cung (last.pt)
    # vao thu muc {project}/{name}/weights/.
    model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        patience=cfg["patience"],
        device=cfg["device"],
        workers=cfg["workers"],
        mosaic=cfg["mosaic"],
        mixup=cfg["mixup"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        fliplr=cfg["fliplr"],
        flipud=cfg["flipud"],
        degrees=cfg["degrees"],
        translate=cfg["translate"],
        scale=cfg["scale"],
        project=project_dir,
        name=cfg["name"],
        exist_ok=cfg["exist_ok"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 cho car/truck/motorbike")
    parser.add_argument("--config", default="train_config.yaml", help="Duong dan file config training")
    args = parser.parse_args()
    train(args.config)
