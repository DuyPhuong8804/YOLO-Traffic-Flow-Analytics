"""So sanh chi so cua mo hinh tren tap train / valid / test -> models/vehicle_yolo26/split_compare.json.

Cung mo hinh, cung cau hinh danh gia (imgsz, conf 0.001, NMS iou 0.7 mac dinh cua Ultralytics); chi doi tap.
Khoang cach train - valid / test cho biet muc do hoc thuoc (overfitting).
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from ultralytics import YOLO  # noqa: E402

from traffic.config import load_config, resolve  # noqa: E402

RUN = ROOT / "models" / "vehicle_yolo26"
DATA = ROOT / "Data" / "Vietnamese vehicle.v3-2023-02-01-5-31pm.yolo26"


def main():
    cfg = load_config()
    model = YOLO(str(resolve(cfg["model"]["weights"])))
    out = {"imgsz": cfg["model"]["imgsz"], "splits": {}}
    for split, key in [("train", "train"), ("valid", "valid"), ("test", "test")]:
        m = model.val(data=cfg["train"]["data"], split={"train": "train", "valid": "val", "test": "test"}[split], imgsz=cfg["model"]["imgsz"], plots=False, verbose=False,
                      project=str(RUN), name="compare_splits_val", exist_ok=True, workers=0)
        per_class = {}
        for i, name in model.names.items():
            p, r, ap50, ap = m.box.class_result(i)
            per_class[name] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50_95": ap}
        out["splits"][split] = {"images": len(list((DATA / key / "images").glob("*"))), "boxes": int(m.nt_per_class.sum()),
                                "precision": m.box.mp, "recall": m.box.mr, "mAP50": m.box.map50,
                                "mAP50_95": m.box.map, "per_class": per_class}
        print(split, {k: round(v, 4) for k, v in out["splits"][split].items() if isinstance(v, float)}, flush=True)
    shutil.rmtree(RUN / "compare_splits_val", ignore_errors=True)
    (RUN / "split_compare.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
