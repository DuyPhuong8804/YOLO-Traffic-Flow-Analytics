"""Do can cu cho cac tham so suy luan / tracking -> models/vehicle_yolo26/param_analysis.json
va hinh docs/images/22_nguong_tin_cay.png.

Gom 3 phan:
  1. Nguong tin cay (conf): duong Precision / Recall / F1 theo conf tren tap valid (chon) va test (kiem tra lai).
  2. Nguong IoU cua NMS: mAP va toc do tren tap test voi iou 0.5 / 0.7 va che do khong NMS (nms=False);
     so box trung (cung lop, IoU > 0.5 voi nhau) tren video Viet Nam.
  3. conf khi tracking: chay ByteTrack tren video Viet Nam voi conf 0.35 va 0.1, dem xe qua vach va so ID.
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from analyze_map import iou_matrix  # noqa: E402
from traffic.classes import CLASS_NAMES  # noqa: E402
from traffic.config import load_config, resolve  # noqa: E402
from traffic.counter import LineCounter  # noqa: E402

RUN = ROOT / "models" / "vehicle_yolo26"
OUT = ROOT / "docs" / "images"
VIDEOS = ["Data/videos/vn_nga_tu_xe_may_o_to.mp4", "Data/videos/vn_ket_xe_xe_may_o_to.mp4",
          "Data/videos/vn_giao_thong_ban_ngay.mp4"]
CONF_POINTS = [0.1, 0.25, 0.35, 0.5]

plt.rcParams.update({"font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight"})


def val(model, data, split, **kw):
    return model.val(data=data, split=split, plots=False, verbose=False, project=str(RUN),
                     name="analyze_params_val", exist_ok=True, **kw)


def smooth(y, f=0.05):
    """Lam tron nhu Ultralytics (box filter) de chon diem F1 lon nhat on dinh."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def conf_curves(m):
    px = m.box.px
    p, r, f1 = m.box.p_curve.mean(0), m.box.r_curve.mean(0), m.box.f1_curve.mean(0)
    best = int(smooth(f1, 0.1).argmax())
    at = {}
    for c in CONF_POINTS:
        i = int(np.searchsorted(px, c))
        at[str(c)] = {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                      "per_class_recall": {CLASS_NAMES[k]: float(m.box.r_curve[k, i]) for k in range(len(CLASS_NAMES))}}
    return {"conf_f1_max": float(px[best]), "f1_max": float(f1[best]), "p_at_f1_max": float(p[best]),
            "r_at_f1_max": float(r[best]), "tai_cac_nguong": at}, (px, p, r, f1, best)


def fig_conf(curves, cfg_conf):
    px, p, r, f1, best = curves
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(px, p, color="#6366F1", lw=2, label="Precision")
    ax.plot(px, r, color="#EC4899", lw=2, label="Recall")
    ax.plot(px, f1, color="#0F172A", lw=2.2, label="F1")
    ax.axvline(px[best], ls=":", color="#64748B", lw=1.4)
    ax.text(px[best] + 0.01, 0.3, f"F1 lớn nhất\nconf = {px[best]:.2f}", ha="left", fontsize=9, color="#475569")
    ax.axvline(cfg_conf, ls="--", color="#EF4444", lw=1.4)
    ax.text(cfg_conf - 0.01, 0.3, f"Ngưỡng dùng trong hệ thống\nconf = {cfg_conf:.2f}", ha="right", fontsize=9,
            color="#EF4444")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Ngưỡng tin cậy (conf)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_title("Precision / Recall / F1 theo ngưỡng tin cậy — tập kiểm định (trung bình 3 lớp)", loc="left",
                 weight="bold", fontsize=11)
    fig.savefig(OUT / "22_nguong_tin_cay.png", facecolor="white")
    plt.close(fig)


def duplicate_stats(model, video, iou, conf, imgsz):
    """Dem so box/khung va so cap box cung lop chong nhau IoU > 0.5 (nghi trung tren cung mot xe)."""
    cap = cv2.VideoCapture(video)
    boxes_total, dup_pairs, frames = 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        xyxy, cls = res.boxes.xyxy.cpu().numpy(), res.boxes.cls.int().cpu().numpy()
        boxes_total += len(xyxy)
        for c in np.unique(cls):
            ious = iou_matrix(xyxy[cls == c], xyxy[cls == c])
            dup_pairs += int(np.triu(ious > 0.5, k=1).sum())
        frames += 1
    cap.release()
    return {"frames": frames, "box_moi_khung": boxes_total / max(frames, 1), "cap_box_trung": dup_pairs}


def tracking_stats(model, video, conf, iou, imgsz, tracker, line_norm):
    """Chay tracking + dem qua vach giong pipeline, tra ve so xe dem duoc va thong ke ID."""
    if getattr(model, "predictor", None) is not None:
        model.predictor.trackers = None
        model.predictor = None
    cap = cv2.VideoCapture(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counter = LineCounter((line_norm[0][0] * w, line_norm[0][1] * h), (line_norm[1][0] * w, line_norm[1][1] * h))
    lengths, frame_idx, boxes_total = Counter(), 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, conf=conf, iou=iou, imgsz=imgsz, tracker=tracker, verbose=False)[0]
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.int().cpu().tolist()
            ids = res.boxes.id.int().cpu().tolist()
            boxes_total += len(ids)
            lengths.update(ids)
            counter.update([(t, CLASS_NAMES[c], ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
                            for b, c, t in zip(boxes, clss, ids)], frame_idx)
        frame_idx += 1
    cap.release()
    lens = np.array(list(lengths.values()))
    return {"frames": frame_idx, "xe_dem_duoc": counter.total, "theo_lop": dict(counter.by_class()),
            "so_id": int(len(lens)), "do_dai_track_trung_vi": float(np.median(lens)) if len(lens) else 0.0,
            "id_ngan_duoi_10_khung": int((lens < 10).sum()), "box_co_id_moi_khung": boxes_total / max(frame_idx, 1)}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=cfg["model"]["weights"])
    args = ap.parse_args()

    model = YOLO(str(resolve(args.weights)))
    data, imgsz = cfg["train"]["data"], cfg["model"]["imgsz"]
    conf, iou, tracker = cfg["model"]["conf"], cfg["model"]["iou"], cfg["model"]["tracker"]
    line = cfg["counting"]["line"]
    out = {"cau_hinh": {"conf": conf, "iou": iou, "imgsz": imgsz, "tracker": tracker}}

    print("1) Nguong tin cay ...")
    conf_valid, curves_valid = conf_curves(val(model, data, "val", imgsz=imgsz))
    conf_test, _ = conf_curves(val(model, data, "test", imgsz=imgsz))
    out["nguong_tin_cay"] = {"valid": conf_valid, "test": conf_test}

    print("2) Nguong IoU cua NMS ...")
    nms_rows = {}
    for name, kw in [("nms_iou_0.5", {"iou": 0.5}), ("nms_iou_0.7", {"iou": 0.7}), ("khong_nms", {"nms": False})]:
        m = val(model, data, "test", imgsz=imgsz, **kw)
        nms_rows[name] = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                          "postprocess_ms": float(m.speed["postprocess"]), "inference_ms": float(m.speed["inference"])}
        print(f"   {name}: mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")
    dup = {}
    for v in VIDEOS[:2]:
        dup[Path(v).name] = {f"iou_{i}": duplicate_stats(model, v, i, conf, imgsz) for i in (0.5, 0.7)}
    out["nms"] = {"tap_test": nms_rows, "box_trung_tren_video": dup}

    print("3) conf khi tracking ...")
    trk = {}
    for v in VIDEOS:
        trk[Path(v).name] = {f"conf_{c}": tracking_stats(model, v, c, iou, imgsz, tracker, line) for c in (conf, 0.1)}
        print("  ", Path(v).name, {k: r["xe_dem_duoc"] for k, r in trk[Path(v).name].items()})
    out["tracking_conf"] = trk

    shutil.rmtree(RUN / "analyze_params_val", ignore_errors=True)
    path = RUN / "param_analysis.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_conf(curves_valid, conf)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("->", path)


if __name__ == "__main__":
    main()
