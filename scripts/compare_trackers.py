"""So sanh ByteTrack va BoT-SORT tren video Viet Nam -> models/vehicle_yolo26/tracker_compare.json.

Cung mo hinh, conf, iou, vach dem; chi doi tracker. Do: so xe dem duoc, so ID, do dai track trung vi,
so ID ngan (< 10 khung, thuong la ID rac / track bi dut) va toc do pipeline (FPS) cua rieng buoc detect + track.
Khong co dem tay nen khong ket luan "dung/sai"; so ID va ID ngan cho biet tracker co bi dut/doi ID hay khong.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from ultralytics import YOLO  # noqa: E402

from analyze_params import tracking_stats  # noqa: E402
from traffic.config import load_config, resolve  # noqa: E402

VIDEOS = ["Data/videos/vn_nga_tu_xe_may_o_to.mp4", "Data/videos/vn_ket_xe_xe_may_o_to.mp4",
          "Data/videos/vn_giao_thong_ban_ngay.mp4", "Data/videos/vn_cao_toc_flycam.mp4"]
TRACKERS = ["bytetrack.yaml", "botsort.yaml"]


def main():
    cfg = load_config()
    mc = cfg["model"]
    model = YOLO(str(resolve(mc["weights"])))
    out = {"conf": mc["conf"], "iou": mc["iou"], "imgsz": mc["imgsz"], "videos": {}}
    for v in VIDEOS:
        out["videos"][Path(v).name] = {}
        for trk in TRACKERS:
            t0 = time.perf_counter()
            r = tracking_stats(model, v, mc["conf"], mc["iou"], mc["imgsz"], trk, cfg["counting"]["line"])
            r["fps"] = r["frames"] / (time.perf_counter() - t0)
            out["videos"][Path(v).name][trk] = r
            print(Path(v).name, trk, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
    path = ROOT / "models" / "vehicle_yolo26" / "tracker_compare.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
