"""Danh gia mo hinh tren tap test + do toc do pipeline tren video mau, ghi ra eval_test.json."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from ultralytics import YOLO  # noqa: E402

from traffic.config import load_config, resolve  # noqa: E402
from traffic.pipeline import TrafficPipeline  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=cfg["model"]["weights"])
    ap.add_argument("--video", default="Data/videos/demo_traffic_sample.mp4")
    args = ap.parse_args()

    weights = resolve(args.weights)
    model = YOLO(str(weights))
    m = model.val(data=cfg["train"]["data"], split="test", imgsz=cfg["model"]["imgsz"], plots=True,
                  project=str(weights.parents[1]), name="eval_test", exist_ok=True, verbose=False)

    per_class = {}
    for i, name in model.names.items():
        p, r, ap50, ap = m.box.class_result(i)
        per_class[name] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50_95": ap}

    fps = None
    if Path(args.video).exists():
        pipe = TrafficPipeline(model=model)
        frames, total_fps = 0, 0.0
        for res in pipe.run(args.video, cfg["counting"]["line"]):
            frames += 1
            if frames > 20:  # bo qua giai doan khoi dong GPU
                total_fps += res.fps
        fps = total_fps / max(frames - 20, 1)

    out = {
        "split": "test",
        "precision": m.box.mp, "recall": m.box.mr, "mAP50": m.box.map50, "mAP50_95": m.box.map,
        "per_class": per_class,
        "speed_ms": m.speed,
        "pipeline_fps": fps,
    }
    path = weights.parents[1] / "eval_test.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
