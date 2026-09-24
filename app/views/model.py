import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from traffic import PROJECT_ROOT
from traffic.classes import CLASS_COLORS_HEX, CLASS_LABELS_VI
from ui import kpi

RUN_DIR = PROJECT_ROOT / "models" / "vehicle_yolo26"
EVAL_DIR = RUN_DIR / "eval_test"
TARGET_MAP = 0.7
TARGET_FPS = 15

st.title("Đánh giá mô hình")
st.caption("YOLO26n fine-tune trên bộ dữ liệu Vietnamese vehicle (Roboflow) — 3 lớp: xe con, xe tải, xe máy.")

results_csv = RUN_DIR / "results.csv"
if not results_csv.exists():
    st.info("Chưa có kết quả huấn luyện. Chạy `python scripts/train.py` để huấn luyện mô hình.")
    st.stop()

df = pd.read_csv(results_csv)
df.columns = [c.strip() for c in df.columns]
best = df.loc[df["metrics/mAP50(B)"].idxmax()]
eval_path = RUN_DIR / "eval_test.json"
ev = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None

# ---------------- KPI ----------------
src = ev or {"mAP50": best["metrics/mAP50(B)"], "mAP50_95": best["metrics/mAP50-95(B)"],
             "precision": best["metrics/precision(B)"], "recall": best["metrics/recall(B)"]}
st.markdown(f"**Chỉ số trên tập {'test' if ev else 'validation (epoch tốt nhất)'}**")
c = st.columns(5)
ok = src["mAP50"] >= TARGET_MAP
with c[0]:
    kpi("mAP@0.5", f"{src['mAP50']:.3f}", "#22C55E" if ok else "#EF4444",
        f"{'Đạt' if ok else 'Chưa đạt'} mục tiêu ≥ {TARGET_MAP}")
with c[1]:
    kpi("mAP@0.5:0.95", f"{src['mAP50_95']:.3f}", "#0EA5E9")
with c[2]:
    kpi("Precision", f"{src['precision']:.3f}", "#6366F1")
with c[3]:
    kpi("Recall", f"{src['recall']:.3f}", "#EC4899")
with c[4]:
    fps = ev.get("pipeline_fps") if ev else None
    if fps:
        kpi("FPS pipeline", f"{fps:.1f}", "#22C55E" if fps >= TARGET_FPS else "#EF4444",
            f"detect + track + đếm · mục tiêu ≥ {TARGET_FPS}")
    else:
        kpi("FPS pipeline", "–", "#94A3B8", "chạy scripts/evaluate.py")

# ---------------- Theo lop ----------------
if ev:
    st.write("")
    st.subheader("Chỉ số theo từng lớp (tập test)")
    rows = [{"Lớp": CLASS_LABELS_VI.get(k, k), **v} for k, v in ev["per_class"].items()]
    pc = pd.DataFrame(rows)
    left, right = st.columns([3, 2], gap="large")
    with left:
        fig = go.Figure()
        for metric, color in [("precision", "#6366F1"), ("recall", "#EC4899"), ("mAP50", "#22C55E"),
                              ("mAP50_95", "#0EA5E9")]:
            fig.add_bar(name=metric.replace("_", "-"), x=pc["Lớp"], y=pc[metric], marker_color=color,
                        text=pc[metric].round(2), textposition="outside")
        fig.add_hline(y=TARGET_MAP, line_dash="dash", line_color="#EF4444",
                      annotation_text=f"mục tiêu mAP {TARGET_MAP}", annotation_position="top right")
        fig.update_layout(barmode="group", height=360, yaxis_range=[0, 1.1], margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
        fig.update_yaxes(gridcolor="#E2E8F0")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.dataframe(pc.set_index("Lớp").style.format("{:.3f}"), width="stretch")
        sp = ev["speed_ms"]
        st.caption(f"Thời gian xử lý / ảnh: tiền xử lý {sp['preprocess']:.1f} ms · suy luận {sp['inference']:.1f} ms"
                   f" · hậu xử lý {sp['postprocess']:.1f} ms")

# ---------------- Duong cong huan luyen ----------------
st.write("")
st.subheader("Quá trình huấn luyện")
left, right = st.columns(2, gap="large")
with left:
    fig = go.Figure()
    for col, name, color in [("metrics/mAP50(B)", "mAP@0.5", "#22C55E"), ("metrics/mAP50-95(B)", "mAP@0.5:0.95", "#0EA5E9"),
                             ("metrics/precision(B)", "Precision", "#6366F1"), ("metrics/recall(B)", "Recall", "#EC4899")]:
        fig.add_scatter(x=df["epoch"], y=df[col], name=name, line=dict(color=color, width=2.5))
    fig.update_layout(height=340, title="Chỉ số validation theo epoch", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig, width="stretch")
with right:
    fig = go.Figure()
    for col, name, color in [("train/box_loss", "box (train)", "#F59E0B"), ("val/box_loss", "box (val)", "#F97316"),
                             ("train/cls_loss", "cls (train)", "#3B82F6"), ("val/cls_loss", "cls (val)", "#1D4ED8")]:
        if col in df:
            fig.add_scatter(x=df["epoch"], y=df[col], name=name, line=dict(color=color, width=2.5))
    fig.update_layout(height=340, title="Hàm mất mát", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig, width="stretch")
st.caption(f"{len(df)} epoch · epoch tốt nhất: {int(best['epoch'])} (mAP@0.5 = {best['metrics/mAP50(B)']:.3f})")

# ---------------- Anh minh hoa ----------------
imgs = [
    ("Ma trận nhầm lẫn (chuẩn hoá)", "confusion_matrix_normalized.png"),
    ("Đường cong Precision–Recall", "BoxPR_curve.png"),
    ("Dự đoán trên ảnh test", "val_batch0_pred.jpg"),
    ("Nhãn thực tế", "val_batch0_labels.jpg"),
]
shown = []
for t, f in imgs:
    p = next((d / f for d in (EVAL_DIR, RUN_DIR) if (d / f).exists()), None)
    if p:
        shown.append((t, p))
if shown:
    st.write("")
    st.subheader("Hình ảnh đánh giá")
    tabs = st.tabs([t for t, _ in shown])
    for tab, (_, p) in zip(tabs, shown):
        tab.image(str(p), width=900)

legend = " · ".join(f"<span style='color:{CLASS_COLORS_HEX[k]}'>■</span> {v}" for k, v in CLASS_LABELS_VI.items())
st.markdown(f"<small>Màu lớp trong toàn hệ thống: {legend}</small>", unsafe_allow_html=True)
