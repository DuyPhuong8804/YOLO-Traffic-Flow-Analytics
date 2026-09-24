"""Do toc do / do chinh xac de chon cau hinh chay thuc te (tuan 9 - toi uu).

1. GPU PyTorch: mAP tap test + FPS pipeline (detect + track + dem) theo kich thuoc anh suy luan.
2. CPU: so sanh PyTorch va ONNX Runtime (kich ban may khong co GPU / thiet bi bien).
Ket qua ghi vao models/vehicle_yolo26/benchmark.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
# Chan Ultralytics tu cai onnxruntime-gpu (xung dot voi onnxruntime); benchmark ONNX chi can ban CPU
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from traffic.config import load_config, resolve  # noqa: E402
from traffic.pipeline import TrafficPipeline  # noqa: E402

WARMUP = 20


def pipeline_fps(model, video, imgsz, frames, device=None) -> float:
    pipe = TrafficPipeline(model=model, imgsz=imgsz, device=device)
    n, t0 = 0, None
    for _ in pipe.run(video, load_config()["counting"]["line"], max_frames=frames + WARMUP):
        n += 1
        if n == WARMUP:
            t0 = time.perf_counter()
    return (n - WARMUP) / (time.perf_counter() - t0)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=cfg["model"]["weights"])
    ap.add_argument("--video", default="Data/videos/vn_giao_thong_ban_ngay.mp4")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--sizes", type=int, nargs="+", default=[320, 480, 640, 960, 1280])
    args = ap.parse_args()
    weights = resolve(args.weights)
    out = {"gpu": None, "gpu_rows": [], "cpu_rows": [], "cpu": os.environ.get("PROCESSOR_IDENTIFIER", "")}

    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        for sz in args.sizes:
            model = YOLO(str(weights))
            m = model.val(data=cfg["train"]["data"], split="test", imgsz=sz, plots=False, verbose=False,
                          project=str(ROOT / "logs"), name="bench_val", exist_ok=True)
            fps = pipeline_fps(model, args.video, sz, args.frames)
            row = {"imgsz": sz, "mAP50": m.box.map50, "mAP50_95": m.box.map, "fps": fps}
            out["gpu_rows"].append(row)
            print(row)

    onnx_path = weights.with_suffix(".onnx")
    if not onnx_path.exists():
        YOLO(str(weights)).export(format="onnx", imgsz=640, simplify=True)
    cpu_frames = max(args.frames // 3, 60)
    for name, path in [("PyTorch", weights), ("ONNX Runtime", onnx_path)]:
        fps = pipeline_fps(YOLO(str(path), task="detect"), args.video, 640, cpu_frames, device="cpu")
        row = {"backend": name, "imgsz": 640, "fps": fps}
        out["cpu_rows"].append(row)
        print(row)

    path = weights.parents[1] / "benchmark.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da ghi {path}")


if __name__ == "__main__":
    main()
