"""Phan tich vi sao mAP@0.5:0.95 thap hon mAP@0.5 tren tap test -> models/vehicle_yolo26/map_analysis.json
va 3 hinh docs/images/19_ap_theo_nguong_iou.png, 20_iou_theo_kich_thuoc.png, 21_vi_du_box_xe_may.jpg.

Gom 4 phan:
  1. AP cua tung lop o tung nguong IoU 0.50..0.95 (model.val -> box.all_ap).
  2. Thong ke kich thuoc box nhan theo lop (pixel sau khi letterbox ve imgsz).
  3. Do chinh xac vi tri: voi moi box nhan, IoU lon nhat voi box du doan cung lop, chia theo nhom kich thuoc.
  4. Danh gia lai cung trong so o imgsz lon hon (khong huan luyen lai).
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from traffic.classes import CLASS_COLORS_HEX, CLASS_LABELS_VI, CLASS_NAMES  # noqa: E402
from traffic.config import load_config, resolve  # noqa: E402

RUN = ROOT / "models" / "vehicle_yolo26"
OUT = ROOT / "docs" / "images"
TEST = ROOT / "Data" / "Vietnamese vehicle.v3-2023-02-01-5-31pm.yolo26" / "test"
IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
# Nhom kich thuoc theo canh sqrt(w*h) tinh bang pixel sau letterbox 640 (tuong tu small/medium/large cua COCO)
SIZE_BINS = [(0, 32, "< 32 px"), (32, 64, "32–64 px"), (64, 128, "64–128 px"), (128, 10_000, "≥ 128 px")]

plt.rcParams.update({"font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight"})


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def read_gt(img_path):
    w, h = Image.open(img_path).size
    lbl = TEST / "labels" / (img_path.stem + ".txt")
    rows = [list(map(float, ln.split())) for ln in lbl.read_text().splitlines() if ln.strip()] if lbl.exists() else []
    cls = np.array([int(r[0]) for r in rows], dtype=int)
    xyxy = np.array([[(r[1] - r[3] / 2) * w, (r[2] - r[4] / 2) * h, (r[1] + r[3] / 2) * w, (r[2] + r[4] / 2) * h]
                     for r in rows]).reshape(-1, 4)
    return w, h, cls, xyxy


def size_bin(side):
    return next(name for lo, hi, name in SIZE_BINS if lo <= side < hi)


def localization_by_size(model, imgsz, conf):
    """Moi box nhan -> IoU lon nhat voi box du doan cung lop (0 neu khong co) + canh box (px o imgsz)."""
    records = []
    images = sorted((TEST / "images").glob("*.*"))
    for img_path in images:
        w, h, gcls, gxyxy = read_gt(img_path)
        scale = imgsz / max(w, h)  # letterbox: canh dai ve imgsz
        res = model.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        pxyxy = res.boxes.xyxy.cpu().numpy()
        pcls = res.boxes.cls.int().cpu().numpy()
        for c in range(len(CLASS_NAMES)):
            g, p = gxyxy[gcls == c], pxyxy[pcls == c]
            ious = iou_matrix(g, p)
            best = ious.max(axis=1) if ious.shape[1] else np.zeros(len(g))
            best_j = ious.argmax(axis=1) if ious.shape[1] else [None] * len(g)
            for box, b, j in zip(g, best, best_j):
                side = float(np.sqrt((box[2] - box[0]) * (box[3] - box[1])) * scale)
                records.append({"cls": CLASS_NAMES[c], "side_px": side, "best_iou": float(b), "img": img_path,
                                "gt": box, "pred": p[j] if j is not None else None})
    return records


def summarize_localization(records):
    out = {}
    for cls in CLASS_NAMES:
        out[cls] = {}
        for lo, hi, name in SIZE_BINS:
            ious = np.array([r["best_iou"] for r in records if r["cls"] == cls and lo <= r["side_px"] < hi])
            if len(ious) == 0:
                continue
            out[cls][name] = {
                "n": int(len(ious)),
                "iou_median": float(np.median(ious)),
                "ti_le_iou_ge_0.5": float((ious >= 0.5).mean()),
                "ti_le_iou_ge_0.75": float((ious >= 0.75).mean()),
                "ti_le_iou_ge_0.9": float((ious >= 0.9).mean()),
            }
    return out


def size_stats(records):
    out = {}
    for cls in CLASS_NAMES:
        sides = np.array([r["side_px"] for r in records if r["cls"] == cls])
        out[cls] = {
            "n": int(len(sides)),
            "canh_trung_vi_px": float(np.median(sides)),
            "canh_trung_binh_px": float(sides.mean()),
            **{f"ti_le_{name}": float(np.mean([size_bin(s) == name for s in sides])) for _, _, name in SIZE_BINS},
        }
    return out


def val_at(model, data, imgsz, batch):
    m = model.val(data=data, split="test", imgsz=imgsz, batch=batch, plots=False, verbose=False,
                  project=str(RUN), name=f"analyze_val_{imgsz}", exist_ok=True)
    return m


def fig_ap_vs_iou(all_ap):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, cls in enumerate(CLASS_NAMES):
        ax.plot(IOU_THRESHOLDS, all_ap[i], "-o", color=CLASS_COLORS_HEX[cls], lw=2,
                label=f"{CLASS_LABELS_VI[cls]} (TB {np.mean(all_ap[i]):.3f})")
    ax.plot(IOU_THRESHOLDS, np.mean(all_ap, axis=0), "--", color="#475569", lw=1.6,
            label=f"Trung bình 3 lớp (mAP@0.5:0.95 = {np.mean(all_ap):.3f})")
    ax.set_xticks(IOU_THRESHOLDS)
    ax.set_xlabel("Ngưỡng IoU để tính là phát hiện đúng")
    ax.set_ylabel("AP")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("AP theo từng ngưỡng IoU trên tập kiểm thử — mAP@0.5:0.95 là trung bình 10 điểm này",
                 loc="left", weight="bold", fontsize=11)
    fig.savefig(OUT / "19_ap_theo_nguong_iou.png", facecolor="white")
    plt.close(fig)


def fig_iou_by_size(loc):
    names = [name for _, _, name in SIZE_BINS]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width = 0.26
    for i, cls in enumerate(CLASS_NAMES):
        vals = [loc[cls].get(n, {}).get("iou_median", np.nan) for n in names]
        counts = [loc[cls].get(n, {}).get("n", 0) for n in names]
        bars = ax.bar([p + (i - 1) * width for p in range(len(names))], vals, width, color=CLASS_COLORS_HEX[cls],
                      label=CLASS_LABELS_VI[cls])
        ax.bar_label(bars, labels=[f"{v:.2f}\nn={c}" if c else "" for v, c in zip(vals, counts)], fontsize=8)
    ax.axhline(0.75, ls="--", color="#EF4444", lw=1.1, label="Ngưỡng IoU 0,75")
    ax.set_xticks(range(len(names)), [f"Cạnh box {n}" for n in names])
    ax.set_ylabel("IoU trung vị (box dự đoán vs box nhãn)")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.set_title("Độ khớp vị trí theo kích thước xe (ảnh letterbox 640 px, tập kiểm thử)", loc="left",
                 weight="bold", fontsize=11)
    fig.savefig(OUT / "20_iou_theo_kich_thuoc.png", facecolor="white")
    plt.close(fig)


def crop_tile(r, size=220):
    img = cv2.imread(str(r["img"]))
    x1, y1, x2, y2 = r["gt"]
    pad = 0.6 * max(x2 - x1, y2 - y1)
    h, w = img.shape[:2]
    cx1, cy1, cx2, cy2 = (int(max(0, x1 - pad)), int(max(0, y1 - pad)), int(min(w, x2 + pad)), int(min(h, y2 + pad)))
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)  # nhan: xanh la
    if r["pred"] is not None:
        px1, py1, px2, py2 = map(int, r["pred"])
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 230), 2)  # du doan: do
    tile = cv2.resize(img[cy1:cy2, cx1:cx2], (size, size))
    cv2.rectangle(tile, (0, 0), (92, 24), (0, 0, 0), -1)
    cv2.putText(tile, f"IoU {r['best_iou']:.2f}", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)


def fig_box_examples(records):
    """Hang tren: xe con ngau nhien; hang duoi: xe may co IoU thap nhat (van phat hien duoc, IoU > 0.3)."""
    mid = [r for r in records if 40 <= r["side_px"] < 128]
    rng = np.random.default_rng(0)
    cars = [r for r in mid if r["cls"] == "car"]
    cars = [cars[i] for i in rng.choice(len(cars), 6, replace=False)]
    bikes = sorted([r for r in mid if r["cls"] == "motorbike" and r["best_iou"] > 0.3], key=lambda r: r["best_iou"])[:6]
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.6))
    for ax, r in zip(axes.flat, cars + bikes):
        ax.imshow(crop_tile(r))
        ax.axis("off")
    axes[0, 0].set_title("Xe con (chọn ngẫu nhiên)", loc="left", weight="bold", fontsize=11)
    axes[1, 0].set_title("Xe máy có IoU thấp nhất (vẫn được phát hiện)", loc="left", weight="bold", fontsize=11)
    fig.suptitle("Box nhãn (xanh lá) và box dự đoán (đỏ) — xe cùng cỡ 40–128 px trên tập kiểm thử", x=0.01,
                 ha="left", weight="bold")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(OUT / "21_vi_du_box_xe_may.jpg", facecolor="white", pil_kwargs={"quality": 88})
    plt.close(fig)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=cfg["model"]["weights"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 800, 960, 1280])
    ap.add_argument("--conf", type=float, default=0.25, help="nguong conf khi do IoU theo kich thuoc")
    args = ap.parse_args()

    model = YOLO(str(resolve(args.weights)))
    data = cfg["train"]["data"]
    base = cfg["model"]["imgsz"]

    print("1) AP theo nguong IoU ...")
    m = val_at(model, data, base, 16)
    all_ap = m.box.all_ap  # (nc, 10)
    ap_by_iou = {CLASS_NAMES[i]: dict(zip(map(str, IOU_THRESHOLDS), map(float, all_ap[i]))) for i in range(len(all_ap))}
    ap_by_iou["tat_ca"] = dict(zip(map(str, IOU_THRESHOLDS), map(float, all_ap.mean(axis=0))))

    print("2-3) Kich thuoc box & do khop vi tri ...")
    records = localization_by_size(model, base, args.conf)
    sizes = size_stats(records)
    loc = summarize_localization(records)

    print("4) Danh gia o imgsz lon hon ...")
    by_imgsz = {}
    for s in args.sizes:
        mm = m if s == base else val_at(model, data, s, 8 if s <= 960 else 4)
        by_imgsz[str(s)] = {
            "mAP50": float(mm.box.map50), "mAP50_95": float(mm.box.map),
            "per_class_mAP50_95": {CLASS_NAMES[i]: float(mm.box.class_result(i)[3]) for i in range(len(CLASS_NAMES))},
            "per_class_mAP50": {CLASS_NAMES[i]: float(mm.box.class_result(i)[2]) for i in range(len(CLASS_NAMES))},
            "inference_ms": float(mm.speed["inference"]),
        }
        print(f"   imgsz {s}: mAP50 {mm.box.map50:.3f}  mAP50-95 {mm.box.map:.3f}")
    for d in RUN.glob("analyze_val_*"):  # thu muc tam do model.val tao ra
        shutil.rmtree(d, ignore_errors=True)

    out = {"weights": str(args.weights), "imgsz_goc": base, "ap_theo_nguong_iou": ap_by_iou,
           "kich_thuoc_box": sizes, "do_khop_vi_tri_theo_kich_thuoc": loc, "danh_gia_theo_imgsz": by_imgsz}
    path = RUN / "map_analysis.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    OUT.mkdir(parents=True, exist_ok=True)
    fig_ap_vs_iou(all_ap)
    fig_iou_by_size(loc)
    fig_box_examples(records)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("->", path)


if __name__ == "__main__":
    main()
