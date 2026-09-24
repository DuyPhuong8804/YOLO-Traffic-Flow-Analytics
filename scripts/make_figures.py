"""Sinh hinh minh hoa cho bao cao vao docs/images/ (so do, bieu do du lieu, huan luyen, danh gia, ket qua chay)."""
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from traffic.classes import CLASS_COLORS_BGR, CLASS_COLORS_HEX, CLASS_LABELS_VI, CLASS_NAMES  # noqa: E402
from traffic.pipeline import TrafficPipeline, load_model  # noqa: E402

OUT = ROOT / "docs" / "images"
RUN = ROOT / "models" / "vehicle_yolo26"
DATASET = ROOT / "Data" / "Vietnamese vehicle.v3-2023-02-01-5-31pm.yolo26"
VIDEOS = ROOT / "Data" / "videos"
SPLITS = {"train": "Huấn luyện", "valid": "Kiểm định", "test": "Kiểm thử"}
VI = [CLASS_LABELS_VI[c] for c in CLASS_NAMES]
COLORS = [CLASS_COLORS_HEX[c] for c in CLASS_NAMES]

plt.rcParams.update({"font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white", **({"pil_kwargs": {"quality": 88}} if name.endswith(".jpg") else {}))
    plt.close(fig)
    print("  ", name)


def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none"))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", color="white", fontsize=12, weight="bold")
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", color="white", fontsize=9, wrap=True)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, color="#475569", lw=1.8))


# ---------------- So do ----------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(15.4, 5.2))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    steps = [
        ("Đầu vào", "Video file\nWebcam · RTSP", "#64748B"),
        ("[1] Phát hiện", "YOLO26n fine-tune\n3 lớp xe", "#2563EB"),
        ("[2] Theo dõi", "ByteTrack / BoT-SORT\ngán ID mỗi xe", "#7C3AED"),
        ("[3] Đếm & phân loại", "cắt vạch → +1\nmỗi ID 1 lần", "#DB2777"),
        ("[4] Lưu trữ", "SQLite\nsessions · events", "#EA580C"),
    ]
    w, h, gap, y = 2.35, 1.6, 0.55, 2.9
    for i, (t, s, c) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        box(ax, x, y, w, h, t, s, c)
        if i:
            arrow(ax, (x - gap + 0.02, y + h / 2), (x - 0.02, y + h / 2))
    xs = 0.2 + 4 * (w + gap)
    box(ax, xs - 3.2, 0.4, 2.8, 1.5, "[5] API thống kê", "FastAPI\n/stats · /events · /sessions", "#059669")
    box(ax, xs + 0.4, 0.4, 2.8, 1.5, "[6] Dashboard", "Streamlit + Plotly\nbiểu đồ · CSV", "#0891B2")
    arrow(ax, (xs + w / 2 - 0.3, y - 0.02), (xs - 1.8, 1.92))
    arrow(ax, (xs + w / 2 + 0.3, y - 0.02), (xs + 1.8, 1.92))
    ax.text(0.2, 0.9, "Ứng dụng: cán bộ quản lý giao thông,\nđơn vị thống kê, quản lý bãi xe,\nnghiên cứu quy hoạch",
            fontsize=10, color="#334155", va="center")
    save(fig, "01_kien_truc_he_thong.png")


def fig_gantt():
    tasks = [
        ("Chuẩn bị: phạm vi, môi trường, khảo sát", 1, 1), ("Chuẩn bị & chuẩn hoá dữ liệu", 2, 2),
        ("Huấn luyện YOLO26 (lần 1)", 3, 3), ("Đánh giá & tinh chỉnh (dự phòng)", 4, 4),
        ("Tracking & logic đếm", 5, 5), ("Lưu trữ SQLite & API", 6, 6), ("Dashboard trực quan", 7, 7),
        ("Kiểm thử & tích hợp end-to-end", 8, 8), ("Tối ưu tốc độ, tài liệu", 9, 9), ("Báo cáo & demo", 10, 10),
    ]
    palette = ["#64748B", "#0EA5E9", "#2563EB", "#2563EB", "#7C3AED", "#EA580C", "#0891B2", "#DB2777", "#059669",
               "#0F172A"]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, ((name, s, e), c) in enumerate(zip(tasks, palette)):
        ax.barh(i, e - s + 1, left=s - 0.5, color=c, height=0.62)
        ax.text(s + 0.55, i, name, va="center", fontsize=10)
    ax.set_yticks([])
    ax.set_xticks(range(1, 11), [f"Tuần {i}" for i in range(1, 11)])
    ax.set_xlim(0.4, 12.4)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E2E8F0")
    ax.set_title("Kế hoạch thực hiện 10 tuần", loc="left", weight="bold")
    save(fig, "02_ke_hoach_10_tuan.png")


def fig_counting_logic():
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(10, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 10, 10, fc="#F1F5F9", ec="#CBD5E1"))
    ax.plot([0.8, 8.2], [6, 6], color="#EF4444", lw=4)
    ax.text(8.35, 6, "B", va="center", weight="bold")
    ax.text(0.2, 6, "A", va="center", weight="bold")
    ax.annotate("", (4.6, 7.3), (4.6, 6), arrowprops=dict(arrowstyle="-|>", color="#F59E0B", lw=2.5))
    ax.text(4.75, 7.1, "chiều VÀO\n(cross(AB, AP) > 0)", color="#B45309", fontsize=9)
    # xe 1: di xuong, cat vach -> dem
    pts = [(3, 2), (3.1, 3.4), (3.2, 4.8), (3.3, 6.4), (3.4, 7.8)]
    ax.plot(*zip(*pts), "-o", color=CLASS_COLORS_HEX["car"], lw=2)
    ax.text(1.0, 1.5, "Xe con #12: đoạn P(t-1)→P(t)\ncắt AB → đếm +1 (Vào)", fontsize=9, color="#1E40AF")
    # xe 2: rung tren vach -> chi dem 1 lan
    pts2 = [(6.2, 5.2), (6.3, 6.5), (6.4, 5.7), (6.5, 6.6), (6.6, 5.6), (6.7, 6.8)]
    ax.plot(*zip(*pts2), "-o", color=CLASS_COLORS_HEX["motorbike"], lw=2)
    ax.text(5.4, 3.6, "Xe máy #31: dao động trên vạch\n→ chỉ đếm 1 lần (đã có ID)", fontsize=9, color="#047857")
    # xe 3: ngoai doan vach
    ax.plot([9.2, 9.3, 9.4], [4, 6, 8], "-o", color=CLASS_COLORS_HEX["truck"], lw=2)
    ax.text(5.6, 9.3, "Xe tải #7: ngoài đoạn AB → không đếm", fontsize=9, color="#B45309")
    ax.set_title("Logic đếm theo vạch ảo (line crossing)", weight="bold")
    save(fig, "08_logic_dem_vach.png")


def fig_db_schema():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    tables = {
        "sessions": (0.3, ["id  INTEGER PK", "source  TEXT", "started_at  TEXT", "ended_at  TEXT", "frames  INTEGER",
                           "avg_fps  REAL", "line  TEXT", "model  TEXT"]),
        "vehicle_events": (5.6, ["id  INTEGER PK", "session_id  INTEGER FK", "track_id  INTEGER", "class_name  TEXT",
                                 "direction  TEXT (in/out)", "frame_idx  INTEGER", "ts  TEXT (thời điểm)"]),
    }
    for name, (x, cols) in tables.items():
        h = 0.42 * (len(cols) + 1)
        y0 = 4.4 - h
        ax.add_patch(plt.Rectangle((x, y0), 4.1, h, fc="white", ec="#334155", lw=1.5))
        ax.add_patch(plt.Rectangle((x, 4.4 - 0.42), 4.1, 0.42, fc="#1E3A8A"))
        ax.text(x + 0.15, 4.4 - 0.21, name, color="white", va="center", weight="bold")
        for i, c in enumerate(cols):
            ax.text(x + 0.15, 4.4 - 0.42 * (i + 1.5), c, va="center", fontsize=10, family="monospace")
    arrow(ax, (5.58, 3.35), (4.42, 3.77))
    ax.text(4.6, 3.05, "1 — n", fontsize=10, color="#475569")
    save(fig, "11_csdl_schema.png")


# ---------------- Du lieu ----------------
def read_labels(split):
    cnt, files = Counter(), sorted((DATASET / split / "labels").glob("*.txt"))
    for f in files:
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                cnt[int(ln.split()[0])] += 1
    return cnt, files


def fig_dataset():
    data = {SPLITS[s]: read_labels(s)[0] for s in SPLITS}
    df = pd.DataFrame({k: [v[i] for i in range(3)] for k, v in data.items()}, index=VI)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [2, 1]})
    x = range(len(df.columns))
    for i, (cls, color) in enumerate(zip(df.index, COLORS)):
        bars = axes[0].bar([p + (i - 1) * 0.26 for p in x], df.loc[cls], 0.26, color=color, label=cls)
        axes[0].bar_label(bars, fontsize=9)
    axes[0].set_xticks(list(x), df.columns)
    axes[0].set_ylabel("Số bounding box")
    axes[0].legend(frameon=False)
    axes[0].set_title("Phân bố nhãn theo tập và lớp", loc="left", weight="bold")
    tot = df.sum(axis=1)
    axes[1].pie(tot, labels=[f"{k}\n{v:,}" for k, v in tot.items()], colors=COLORS, autopct="%1.0f%%",
                wedgeprops=dict(width=0.45), pctdistance=0.78, textprops=dict(fontsize=10))
    axes[1].set_title(f"Tổng {tot.sum():,} box", weight="bold")
    save(fig, "03_du_lieu_phan_bo_lop.png")


def draw_gt(img_path, lbl_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for ln in lbl_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        c, cx, cy, bw, bh = ln.split()
        c, cx, cy, bw, bh = int(c), float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)),
                      CLASS_COLORS_BGR[CLASS_NAMES[c]], 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fig_dataset_samples():
    random.seed(7)
    _, files = read_labels("train")
    picks = random.sample([f for f in files if len(f.read_text().splitlines()) >= 3], 8)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    for ax, lbl in zip(axes.flat, picks):
        img = next((DATASET / "train" / "images").glob(lbl.stem + ".*"))
        ax.imshow(draw_gt(img, lbl))
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, VI, loc="lower center", ncol=3, frameon=False, fontsize=12)
    fig.suptitle("Ảnh mẫu tập huấn luyện kèm nhãn (đã chuẩn hoá 3 lớp)", weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, "04_du_lieu_anh_mau.jpg")


# ---------------- Huan luyen & danh gia ----------------
def fig_training():
    df = pd.read_csv(RUN / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for col, name, c in [("metrics/mAP50(B)", "mAP@0.5", "#22C55E"), ("metrics/mAP50-95(B)", "mAP@0.5:0.95", "#0EA5E9"),
                         ("metrics/precision(B)", "Precision", "#6366F1"), ("metrics/recall(B)", "Recall", "#EC4899")]:
        axes[0].plot(df["epoch"], df[col], label=name, color=c, lw=2)
    axes[0].axhline(0.7, ls="--", color="#EF4444", lw=1.2)
    axes[0].text(df["epoch"].max(), 0.705, "mục tiêu mAP 0.7", color="#EF4444", ha="right", va="bottom", fontsize=9)
    axes[0].set_title("Chỉ số trên tập kiểm định theo epoch", loc="left", weight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(color="#E2E8F0")
    for col, name, c, ls in [("train/box_loss", "box (train)", "#F59E0B", "-"), ("val/box_loss", "box (val)", "#F59E0B", "--"),
                             ("train/cls_loss", "cls (train)", "#2563EB", "-"), ("val/cls_loss", "cls (val)", "#2563EB", "--")]:
        axes[1].plot(df["epoch"], df[col], label=name, color=c, ls=ls, lw=2)
    axes[1].set_title("Hàm mất mát", loc="left", weight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(color="#E2E8F0")
    save(fig, "05_huan_luyen_duong_cong.png")


def fig_eval():
    ev = json.loads((RUN / "eval_test.json").read_text(encoding="utf-8"))
    metrics = [("precision", "Precision", "#6366F1"), ("recall", "Recall", "#EC4899"), ("mAP50", "mAP@0.5", "#22C55E"),
               ("mAP50_95", "mAP@0.5:0.95", "#0EA5E9")]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, (k, name, c) in enumerate(metrics):
        vals = [ev["per_class"][cls][k] for cls in CLASS_NAMES] + [ev[k]]
        bars = ax.bar([p + (i - 1.5) * 0.2 for p in range(4)], vals, 0.2, color=c, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.axhline(0.7, ls="--", color="#EF4444", lw=1.2)
    ax.set_xticks(range(4), VI + ["Tất cả"])
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.set_title("Kết quả đánh giá trên tập kiểm thử (156 ảnh, 729 box)", loc="left", weight="bold")
    save(fig, "06_danh_gia_theo_lop.png")
    for src, dst in [("eval_test/confusion_matrix_normalized.png", "07a_ma_tran_nham_lan.png"),
                     ("eval_test/BoxPR_curve.png", "07b_duong_cong_pr.png"),
                     ("eval_test/val_batch0_pred.jpg", "07c_du_doan_tap_test.jpg")]:
        shutil.copy(RUN / src, OUT / dst)
        print("  ", dst)


def fig_benchmark():
    path = RUN / "benchmark.json"
    if not path.exists():
        print("   (bo qua benchmark: chua co benchmark.json)")
        return
    b = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(b["gpu_rows"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [3, 2], "wspace": 0.4})
    ax = axes[0]
    bars = ax.bar(rows["imgsz"].astype(str), rows["fps"], color="#93C5FD", label="FPS pipeline")
    ax.bar_label(bars, fmt="%.1f", fontsize=9)
    ax.axhline(15, ls="--", color="#EF4444", lw=1.2)
    ax.text(4.4, 15.8, "mục tiêu 15 FPS", color="#EF4444", fontsize=9, ha="right")
    ax.set_ylim(0, max(rows["fps"]) * 1.2)
    ax.set_xlabel("Kích thước ảnh suy luận (imgsz)")
    ax.set_ylabel("FPS (detect + track + đếm)")
    ax2 = ax.twinx()
    ax2.plot(rows["imgsz"].astype(str), rows["mAP50"], "-o", color="#16A34A", lw=2, label="mAP@0.5 (test)")
    ax2.set_ylim(0.5, 1.0)
    ax2.set_ylabel("mAP@0.5")
    ax2.spines["right"].set_visible(True)
    ax.set_title(f"Tốc độ – độ chính xác theo imgsz ({b['gpu']})", loc="left", weight="bold", fontsize=11)
    fig.legend(loc="lower center", bbox_to_anchor=(0.33, -0.06), ncol=2, frameon=False)
    cpu = pd.DataFrame(b["cpu_rows"])
    bars = axes[1].bar(cpu["backend"], cpu["fps"], color=["#94A3B8", "#F97316"], width=0.5)
    axes[1].bar_label(bars, fmt="%.1f FPS", fontsize=10)
    axes[1].set_title("Chạy trên CPU (imgsz 640)", loc="left", weight="bold")
    axes[1].set_ylabel("FPS")
    save(fig, "12_toi_uu_toc_do.png")


# ---------------- Ket qua chay tren video ----------------
def grab_frames(model, video, line, frame_ids):
    pipe = TrafficPipeline(model=model)
    got = {}
    for res in pipe.run(str(video), line, max_frames=max(frame_ids) + 1):
        if res.frame_idx - 1 in frame_ids:
            got[res.frame_idx - 1] = (cv2.cvtColor(res.frame, cv2.COLOR_BGR2RGB), res.total)
    return [got[i] for i in frame_ids]


def fig_tracking(model):
    frames = grab_frames(model, VIDEOS / "demo_traffic_sample.mp4", [[0.05, 0.6], [0.95, 0.6]], [396, 408, 420, 432])
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.0))
    for ax, (img, total), i in zip(axes, frames, [396, 408, 420, 432]):
        ax.imshow(img)
        ax.set_title(f"Khung hình {i}  (đã đếm: {total})", fontsize=10)
        ax.axis("off")
    fig.suptitle("Theo dõi đối tượng: mỗi xe giữ nguyên ID qua các khung hình, đổi trạng thái khi cắt vạch",
                 weight="bold")
    fig.tight_layout()
    save(fig, "09_theo_doi_id.jpg")


def fig_vn_videos(model):
    cases = [
        ("vn_giao_thong_ban_ngay.mp4", [[0.05, 0.6], [0.95, 0.6]], 0.7, "Giao lộ ban ngày — nhận tốt xe máy; SUV cận cảnh bị nhầm thành xe tải"),
        ("vn_hcm_nga_tu_ban_dem.mp4", [[0.05, 0.6], [0.95, 0.6]], 0.6, "Ngã tư TP.HCM ban đêm — biển quảng cáo sáng bị nhầm thành xe tải"),
        ("vn_ket_xe_xe_may_o_to.mp4", [[0.05, 0.6], [0.95, 0.6]], 0.6, "Kẹt xe — bỏ sót nhiều xe máy nhỏ, dày đặc"),
        ("vn_cao_toc_flycam.mp4", [[0.05, 0.6], [0.95, 0.6]], 0.6, "Flycam (ngoài phạm vi) — nhận nhầm mái nhà"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.4))
    for ax, (name, line, pos, title) in zip(axes.flat, cases):
        n = int(cv2.VideoCapture(str(VIDEOS / name)).get(cv2.CAP_PROP_FRAME_COUNT))
        (img, _), = grab_frames(model, VIDEOS / name, line, [int(n * pos)])
        ax.imshow(img)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.axis("off")
    fig.tight_layout()
    save(fig, "10_ket_qua_video_viet_nam.jpg")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("So do:")
    fig_architecture()
    fig_gantt()
    fig_counting_logic()
    fig_db_schema()
    print("Du lieu:")
    fig_dataset()
    fig_dataset_samples()
    print("Huan luyen & danh gia:")
    fig_training()
    fig_eval()
    fig_benchmark()
    print("Ket qua chay:")
    model = load_model()
    fig_tracking(model)
    fig_vn_videos(model)


if __name__ == "__main__":
    main()
