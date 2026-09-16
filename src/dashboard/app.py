"""
Dashboard Streamlit thong ke luu luong giao thong.

Gom 2 tab doc lap:
    - "Video truc tiep": chay TrafficPipeline TRUC TIEP trong tien trinh Streamlit
      de xem preview video co bbox/line/bo dem.
    - "Thong ke": doc TRUC TIEP tu SQLite (khong qua API trung gian) va ve bieu do.

Chay:
    streamlit run src/dashboard/app.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config_loader import load_yaml  # noqa: E402
from src.pipeline import TrafficPipeline  # noqa: E402
from src.storage import stats  # noqa: E402

st.set_page_config(page_title="Traffic Flow Analytics", layout="wide")
st.title("📊 Hệ thống giám sát & thống kê lưu lượng giao thông")

tab_live, tab_stats = st.tabs(["🎥 Video trực tiếp", "📈 Thống kê"])


# ============================== TAB 1: VIDEO TRUC TIEP ==============================
with tab_live:
    lines_cfg = load_yaml("counting_lines.yaml")
    camera_ids = list(lines_cfg["cameras"].keys())
    camera_id = st.selectbox("Chọn camera", camera_ids)

    col_start, col_stop = st.columns(2)
    start_clicked = col_start.button("▶️ Bắt đầu xử lý", use_container_width=True)
    stop_clicked = col_stop.button("⏹️ Dừng", use_container_width=True)

    if start_clicked:
        st.session_state["live_running"] = True
    if stop_clicked:
        st.session_state["live_running"] = False

    video_placeholder = st.empty()
    metrics_placeholder = st.empty()

    if st.session_state.get("live_running"):
        pipeline = TrafficPipeline(camera_id=camera_id, draw_overlay=True)
        for result in pipeline.run():
            # Streamlit chay script tren 1 thread; kiem tra lai session_state moi vong lap
            # de cho phep nguoi dung bam nut "Dung" tu 1 lan rerun khac kip dung vong lap nay.
            if not st.session_state.get("live_running"):
                break

            frame_rgb = result.frame[:, :, ::-1]  # BGR (OpenCV) -> RGB (Streamlit)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            with metrics_placeholder.container():
                cols = st.columns(max(len(result.cumulative_counts), 1) + 1)
                cols[0].metric("FPS", f"{result.fps:.1f}")
                for i, (label, value) in enumerate(result.cumulative_counts.items(), start=1):
                    if i < len(cols):
                        cols[i].metric(label, value)


# ============================== TAB 2: THONG KE ==============================
with tab_stats:
    app_cfg = load_yaml("app_config.yaml")
    db_path = app_cfg["db_path"]

    def _get_connection() -> sqlite3.Connection:
        return sqlite3.connect(db_path)

    @st.fragment(run_every="10s")
    def render_stats() -> None:
        conn = _get_connection()
        try:
            col1, col2 = st.columns(2)

            # ---- Bieu do luu luong theo gio (line chart) ----
            hourly = stats.get_hourly_stats(conn, limit_hours=24)
            with col1:
                st.subheader("Lưu lượng theo giờ (24h gần nhất)")
                if hourly:
                    df = pd.DataFrame(hourly)
                    pivot = df.pivot_table(index="hour", columns="loai_xe", values="count", fill_value=0)
                    st.line_chart(pivot)
                else:
                    st.info("Chưa có dữ liệu.")

            # ---- Bieu do phan bo theo loai xe (bar chart) ----
            by_type = stats.get_stats_by_type(conn)
            with col2:
                st.subheader("Phân bố theo loại xe")
                if by_type:
                    df = pd.DataFrame(by_type).set_index("loai_xe")
                    st.bar_chart(df)
                else:
                    st.info("Chưa có dữ liệu.")

            # ---- Bang gio cao diem ----
            st.subheader("Giờ cao điểm")
            peak = stats.get_peak_hours(conn, top_n=5)
            if peak:
                st.dataframe(pd.DataFrame(peak), use_container_width=True)
            else:
                st.info("Chưa có dữ liệu.")

            # ---- Luu luong theo ngay ----
            st.subheader("Lưu lượng theo ngày (30 ngày gần nhất)")
            daily = stats.get_daily_stats(conn, limit_days=30)
            if daily:
                df = pd.DataFrame(daily)
                pivot = df.pivot_table(index="date", columns="loai_xe", values="count", fill_value=0)
                st.bar_chart(pivot)
            else:
                st.info("Chưa có dữ liệu.")
        finally:
            conn.close()

    render_stats()
