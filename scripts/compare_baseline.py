"""So sanh mo hinh fine-tune voi mo hinh COCO pretrained (cach da so du an ma nguon mo tren mang dang dung)
-> models/vehicle_yolo26/baseline_compare.json va hinh docs/images/23_so_sanh_baseline.png.

Hai phan, cung tap test, cung cach tinh mAP (ap_per_class cua Ultralytics, conf thap 0.001, NMS iou 0.7):
  1. mAP tren tap test 3 lop. COCO: car->car, truck + bus->truck, motorcycle->motorbike (giong COCO_TO_PROJECT).
  2. Tren video Viet Nam: so box/khung va so xe dem qua vach (cung tracker, cung vach) cua 2 mo hinh.
"""
import argparse
import json
import os
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
import torch  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.metrics import ap_per_class, box_iou  # noqa: E402

from traffic.classes import CLASS_NAMES, to_project_class  # noqa: E402
from traffic.config import load_config, resolve  # noqa: E402
from traffic.counter import LineCounter  # noqa: E402

RUN = ROOT / "models" / "vehicle_yolo26"
OUT = ROOT / "docs" / "images"
VIDEOS = ["Data/videos/vn_giao_thong_ban_ngay.mp4", "Data/videos/vn_nga_tu_xe_may_o_to.mp4",
          "Data/videos/vn_hcm_nga_tu_ban_dem.mp4", "Data/videos/vn_ket_xe_xe_may_o_to.mp4"]
IOU_THRS = torch.linspace(0.5, 0.95, 10)

plt.rcParams.update({"font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight"})


def class_mapper(model):
    """id lop cua mo hinh -> id lop du an (hoac -1 neu bo)."""
    return {i: (CLASS_NAMES.index(p) if (p := to_project_class(n)) else -1) for i, n in model.names.items()}


def load_gt(label_path, w, h):
    rows = [list(map(float, ln.split())) for ln in label_path.read_text().splitlines() if ln.strip()] \
        if label_path.exists() else []
    if not rows:
        return torch.zeros(0), torch.zeros((0, 4))
    a = np.array(rows)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return torch.tensor(a[:, 0]).long(), torch.tensor(xyxy, dtype=torch.float32)


def match(pred_cls, pred_box, gt_cls, gt_box):
    """Ma tran dung/sai (n_pred x 10 nguong IoU) nhu DetectionValidator.match_predictions."""
    correct = np.zeros((len(pred_cls), len(IOU_THRS)), dtype=bool)
    if len(gt_cls) == 0 or len(pred_cls) == 0:
        return correct
    iou = box_iou(gt_box, pred_box).numpy() * (gt_cls[:, None] == pred_cls[None, :]).numpy()
    for k, thr in enumerate(IOU_THRS.tolist()):
        gi, pi = np.nonzero(iou >= thr)
        if len(gi):
            m = np.concatenate([np.stack([gi, pi], 1), iou[gi, pi][:, None]], 1)
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), k] = True
    return correct


def eval_test(model, img_dir, label_dir, imgsz):
    cmap = class_mapper(model)
    tp, conf, pcls, tcls = [], [], [], []
    for img in sorted(img_dir.glob("*")):
        frame = cv2.imread(str(img))
        h, w = frame.shape[:2]
        res = model.predict(frame, conf=0.001, iou=0.7, imgsz=imgsz, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu()
        cls = torch.tensor([cmap[int(c)] for c in res.boxes.cls.cpu()]).long()
        sc = res.boxes.conf.cpu()
        keep = cls >= 0
        xyxy, cls, sc = xyxy[keep], cls[keep], sc[keep]
        gcls, gbox = load_gt(label_dir / (img.stem + ".txt"), w, h)
        tp.append(match(cls, xyxy, gcls, gbox))
        conf.append(sc.numpy())
        pcls.append(cls.numpy())
        tcls.append(gcls.numpy())
    tp, conf, pcls, tcls = map(np.concatenate, (tp, conf, pcls, tcls))
    r = ap_per_class(tp, conf, pcls, tcls)
    ap = r[5]  # (n_class, 10)
    present = r[6].astype(int)
    out = {"mAP50": float(ap[:, 0].mean()), "mAP50_95": float(ap.mean())}
    out["per_class"] = {CLASS_NAMES[c]: {"mAP50": float(ap[i, 0]), "mAP50_95": float(ap[i].mean())}
                        for i, c in enumerate(present)}
    return out


def video_stats(model, video, conf, iou, imgsz, tracker, line):
    cmap = class_mapper(model)
    model.predictor = None
    cap = cv2.VideoCapture(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counter = LineCounter((line[0][0] * w, line[0][1] * h), (line[1][0] * w, line[1][1] * h))
    frames, boxes, ids = 0, 0, set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, conf=conf, iou=iou, imgsz=imgsz, tracker=tracker, verbose=False)[0]
        if res.boxes is not None and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            dets = []
            for b, c, t in zip(xyxy, res.boxes.cls.int().cpu().tolist(), res.boxes.id.int().cpu().tolist()):
                if cmap[c] >= 0:
                    dets.append((t, CLASS_NAMES[cmap[c]], ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)))
            boxes += len(dets)
            ids.update(d[0] for d in dets)
            counter.update(dets, frames)
        frames += 1
    cap.release()
    return {"frames": frames, "box_moi_khung": boxes / max(frames, 1), "so_id": len(ids),
            "xe_dem_duoc": counter.total, "theo_lop": dict(counter.by_class())}


def fig_compare(res):
    names = ["Tổng (mAP@0.5)", "Xe con", "Xe tải", "Xe máy"]
    keys = [None, "car", "truck", "motorbike"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in zip(axes, ["mAP50", "mAP50_95"], ["mAP@0.5", "mAP@0.5:0.95"]):
        x = np.arange(4)
        for j, (label, color) in enumerate([("coco_pretrained", "#94A3B8"), ("fine_tune", "#6366F1")]):
            vals = [res["tap_test"][label][metric] if k is None else res["tap_test"][label]["per_class"][k][metric]
                    for k in keys]
            bars = ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=color,
                          label="COCO pretrained (không huấn luyện lại)" if j == 0 else "Fine-tune (dự án này)")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x, ["Tổng"] + names[1:])
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", weight="bold", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.1, -0.1), ncol=2)
    fig.savefig(OUT / "23_so_sanh_baseline.png", facecolor="white")
    plt.close(fig)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=cfg["model"]["weights"])
    ap.add_argument("--baseline", default=cfg["model"]["fallback_weights"])
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()

    ds = yaml.safe_load((ROOT / cfg["train"]["data"]).read_text(encoding="utf-8"))
    base = ROOT / ds["path"] / ds["test"]
    img_dir, label_dir = base, base.parent / "labels"
    imgsz, conf, iou, tracker = cfg["model"]["imgsz"], cfg["model"]["conf"], cfg["model"]["iou"], cfg["model"]["tracker"]
    models = {"coco_pretrained": YOLO(str(resolve(args.baseline))), "fine_tune": YOLO(str(resolve(args.weights)))}

    out = {"tap_test": {}, "video": {}}
    for name, m in models.items():
        out["tap_test"][name] = eval_test(m, img_dir, label_dir, imgsz)
        print(name, {k: round(v, 3) for k, v in out["tap_test"][name].items() if k.startswith("mAP")})
    if not args.skip_video:
        for v in VIDEOS:
            if not Path(v).exists():
                continue
            out["video"][Path(v).name] = {n: video_stats(m, v, conf, iou, imgsz, tracker, cfg["counting"]["line"])
                                          for n, m in models.items()}
            print(Path(v).name, {n: (r["box_moi_khung"], r["xe_dem_duoc"]) for n, r in out["video"][Path(v).name].items()})
    (RUN / "baseline_compare.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_compare(out)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
