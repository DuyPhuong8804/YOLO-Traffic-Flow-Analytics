from datetime import datetime, time as dtime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from traffic.classes import CLASS_COLORS_HEX, CLASS_LABELS_VI, CLASS_NAMES
from ui import class_kpis, get_db, kpi

LABEL_COLORS = {CLASS_LABELS_VI[c]: CLASS_COLORS_HEX[c] for c in CLASS_NAMES}
INTERVAL_LABELS = {"auto": "Tự động", "10s": "Theo 10 giây", "minute": "Theo phút", "hour": "Theo giờ",
                   "day": "Theo ngày"}
FREQ = {"10s": "10s", "minute": "min", "hour": "h", "day": "D"}
UNIT = {"10s": "10 giây", "minute": "phút", "hour": "giờ", "day": "ngày"}
PLOT_LAYOUT = dict(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)",
                   paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2, title=None),
                   font=dict(size=13))

db = get_db()
st.title("Dashboard thống kê")
st.caption("Tổng hợp lưu lượng phương tiện từ tất cả các phiên phân tích đã lưu.")

sessions = db.sessions()
if sessions.empty:
    st.info("Chưa có dữ liệu. Hãy chạy một phiên phân tích video trước.", icon=":material/info:")
    st.page_link("views/analyze.py", label="Đi tới Phân tích video", icon=":material/videocam:")
    st.stop()

# ---------------- Bo loc ----------------
tmin, tmax = db.time_range()
with st.container(border=True):
    f1, f2, f3 = st.columns([2, 2, 1])
    session_opts = {0: "Tất cả phiên"} | {
        int(r.id): f"#{r.id} · {r.source} · {r.started_at[:16]} ({r.vehicles} xe)" for r in sessions.itertuples()
    }
    sid = f1.selectbox("Phiên phân tích", list(session_opts), format_func=session_opts.get) or None
    if tmin:
        d0, d1 = pd.to_datetime(tmin).date(), pd.to_datetime(tmax).date()
        rng = f2.date_input("Khoảng ngày", (d0, d1), min_value=d0, max_value=d1)
    else:
        rng = ()
    interval = f3.selectbox("Gom nhóm", list(INTERVAL_LABELS), format_func=INTERVAL_LABELS.get)

start = datetime.combine(rng[0], dtime.min) if len(rng) >= 1 else None
end = datetime.combine(rng[1] if len(rng) == 2 else rng[0], dtime.max).replace(microsecond=0) if rng else None

events = db.events(start, end, sid)
if events.empty:
    st.warning("Không có phương tiện nào trong bộ lọc đã chọn.")
    st.stop()

events["ts"] = pd.to_datetime(events["ts"])
events["Loại xe"] = events["class_name"].map(CLASS_LABELS_VI)
events["Chiều"] = events["direction"].map({"in": "Vào", "out": "Ra"})

if interval == "auto":
    span = events["ts"].max() - events["ts"].min()
    interval = ("10s" if span <= pd.Timedelta(minutes=15) else "minute" if span <= pd.Timedelta(hours=3)
                else "hour" if span <= pd.Timedelta(days=3) else "day")

# ---------------- KPI ----------------
by_class = events["class_name"].value_counts().to_dict()
total = len(events)
class_kpis(by_class, total, {"total": f"{events['ts'].min():%d/%m %H:%M} → {events['ts'].max():%d/%m %H:%M}"})
st.write("")
c1, c2, c3, c4 = st.columns(4)
n_in = int((events.direction == "in").sum())
with c1:
    kpi("Chiều vào", f"{n_in:,}", "#6366F1", f"{n_in / total:.0%}")
with c2:
    kpi("Chiều ra", f"{total - n_in:,}", "#EC4899", f"{(total - n_in) / total:.0%}")
freq = FREQ[interval]
per_bucket = events.set_index("ts").resample(freq).size()
unit = UNIT[interval]
with c3:
    kpi(f"Trung bình / {unit}", f"{per_bucket.mean():.1f}", "#0EA5E9")
with c4:
    kpi(f"Cao điểm / {unit}", f"{per_bucket.max():,}", "#F97316", f"lúc {per_bucket.idxmax():%d/%m %H:%M:%S}")

# ---------------- Bieu do ----------------
st.write("")
left, right = st.columns([2, 1], gap="large")
with left:
    st.subheader(f"Lưu lượng {INTERVAL_LABELS[interval].lower()}")
    ts = (events.assign(bucket=events["ts"].dt.floor(freq))
          .groupby(["bucket", "Loại xe"]).size().rename("Số xe").reset_index())
    fig = px.bar(ts, x="bucket", y="Số xe", color="Loại xe", color_discrete_map=LABEL_COLORS,
                 category_orders={"Loại xe": list(LABEL_COLORS)}, labels={"bucket": ""})
    fig.update_layout(**PLOT_LAYOUT, height=380, bargap=0.15)
    fig.update_yaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Cơ cấu phương tiện")
    share = events["Loại xe"].value_counts().reindex(list(LABEL_COLORS), fill_value=0)
    share = share[share > 0]
    fig = go.Figure(go.Pie(labels=share.index, values=share.values, hole=0.6, sort=False,
                           marker=dict(colors=[LABEL_COLORS[k] for k in share.index]),
                           textinfo="percent", textfont_size=14))
    fig.update_layout(**PLOT_LAYOUT, height=380,
                      annotations=[dict(text=f"<b>{total:,}</b><br>xe", showarrow=False, font_size=20)])
    st.plotly_chart(fig, width="stretch")

left, right = st.columns([1, 2], gap="large")
with left:
    st.subheader("Theo chiều di chuyển")
    dir_df = events.groupby(["Loại xe", "Chiều"]).size().rename("Số xe").reset_index()
    fig = px.bar(dir_df, x="Loại xe", y="Số xe", color="Chiều", barmode="group",
                 color_discrete_map={"Vào": "#6366F1", "Ra": "#EC4899"},
                 category_orders={"Loại xe": list(LABEL_COLORS), "Chiều": ["Vào", "Ra"]})
    fig.update_layout(**PLOT_LAYOUT, height=340)
    fig.update_yaxes(gridcolor="#E2E8F0")
    fig.update_xaxes(title=None)
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Lưu lượng tích luỹ")
    cum = events.sort_values("ts").copy()
    cum["Tích luỹ"] = cum.groupby("Loại xe").cumcount() + 1
    fig = px.line(cum, x="ts", y="Tích luỹ", color="Loại xe", color_discrete_map=LABEL_COLORS,
                  line_shape="hv", labels={"ts": ""}, category_orders={"Loại xe": list(LABEL_COLORS)})
    fig.update_layout(**PLOT_LAYOUT, height=340)
    fig.update_yaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig, width="stretch")

# ---------------- Bang du lieu ----------------
tab1, tab2 = st.tabs(["Chi tiết phương tiện", "Các phiên phân tích"])
with tab1:
    table = events[["ts", "Loại xe", "Chiều", "track_id", "session_id", "frame_idx"]].rename(columns={
        "ts": "Thời điểm", "track_id": "Track ID", "session_id": "Phiên", "frame_idx": "Khung hình"})
    st.dataframe(table, width="stretch", hide_index=True, height=360,
                 column_config={"Thời điểm": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm:ss")})
    st.download_button("Xuất CSV", table.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"luu_luong_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv",
                       icon=":material/download:")
with tab2:
    view = sessions.rename(columns={
        "id": "Phiên", "source": "Nguồn", "started_at": "Bắt đầu", "ended_at": "Kết thúc",
        "frames": "Số khung hình", "avg_fps": "FPS TB", "vehicles": "Số xe", "model": "Mô hình"})
    st.dataframe(view.drop(columns=["line"]), width="stretch", hide_index=True)
    with st.expander("Xoá phiên"):
        to_del = st.selectbox("Chọn phiên cần xoá", list(session_opts)[1:], format_func=session_opts.get)
        if st.button("Xoá phiên này", type="secondary", icon=":material/delete:"):
            db.delete_session(to_del)
            st.toast(f"Đã xoá phiên #{to_del}")
            st.rerun()
