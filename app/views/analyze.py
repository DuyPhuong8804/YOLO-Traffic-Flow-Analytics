import html
from collections import deque
from datetime import datetime, time as dtime
from pathlib import Path

import cv2
import streamlit as st

from traffic import PROJECT_ROOT
from traffic.classes import CLASS_COLORS_HEX, CLASS_LABELS_VI
from traffic.config import load_config
from traffic.drawing import preview_with_line
from traffic.pipeline import TrafficPipeline, default_weights, load_model, read_first_frame
from ui import get_db, kpi, pill

SAMPLE_DIR = PROJECT_ROOT / "Data" / "videos"
UPLOAD_DIR = PROJECT_ROOT / "Data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "outputs"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
DISPLAY_MAX_W = 960
TRACKERS = {"bytetrack.yaml": "ByteTrack", "botsort.yaml": "BoT-SORT"}


@st.cache_resource(show_spinner="Đang tải mô hình YOLO26...")
def get_model(path: str):
    return load_model(path)


@st.cache_data(show_spinner=False)
def first_frame(source: str):
    return read_first_frame(source)


def to_display(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if w > DISPLAY_MAX_W:
        frame_bgr = cv2.resize(frame_bgr, (DISPLAY_MAX_W, int(h * DISPLAY_MAX_W / w)))
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


st.title("Phân tích video")
st.caption("Chọn nguồn video, đặt vạch đếm rồi bấm **Bắt đầu phân tích**. Kết quả được lưu tự động vào CSDL.")

# ---------------- Sidebar: cau hinh ----------------
with st.sidebar:
    st.header("1. Nguồn video")
    mode = st.radio("Loại nguồn", ["Video mẫu", "Tải video lên", "Webcam", "Camera IP (RTSP)"],
                    label_visibility="collapsed")
    source = None
    if mode == "Video mẫu":
        samples = sorted(p.name for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in VIDEO_EXT)
        if samples:
            source = str(SAMPLE_DIR / st.selectbox("Chọn video", samples))
        else:
            st.info("Chưa có video trong Data/videos")
    elif mode == "Tải video lên":
        up = st.file_uploader("Tệp video", type=[e[1:] for e in VIDEO_EXT])
        if up:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            dest = UPLOAD_DIR / Path(up.name).name
            if not dest.exists() or dest.stat().st_size != up.size:
                dest.write_bytes(up.getbuffer())
            source = str(dest)
    elif mode == "Webcam":
        source = str(st.number_input("Chỉ số webcam", 0, 9, 0))
    else:
        url = st.text_input("Địa chỉ luồng", placeholder="rtsp://user:pass@192.168.1.10:554/stream")
        source = url.strip() or None

    st.header("2. Vạch đếm")
    default_line = load_config()["counting"]["line"]
    kind = st.segmented_control("Kiểu vạch", ["Ngang", "Dọc", "Tuỳ chỉnh"], default="Ngang")
    if kind == "Dọc":
        x = st.slider("Vị trí ngang (%)", 0, 100, 50)
        y1, y2 = st.slider("Phạm vi dọc (%)", 0, 100, (5, 95))
        line = [[x / 100, y1 / 100], [x / 100, y2 / 100]]
    elif kind == "Tuỳ chỉnh":
        c1, c2 = st.columns(2)
        ax, bx = c1.slider("A · x (%)", 0, 100, 5), c1.slider("B · x (%)", 0, 100, 95)
        ay, by = c2.slider("A · y (%)", 0, 100, 60), c2.slider("B · y (%)", 0, 100, 60)
        line = [[ax / 100, ay / 100], [bx / 100, by / 100]]
    else:
        y = st.slider("Vị trí dọc (%)", 0, 100, int(default_line[0][1] * 100))
        x1, x2 = st.slider("Phạm vi ngang (%)", 0, 100,
                           (int(default_line[0][0] * 100), int(default_line[1][0] * 100)))
        line = [[x1 / 100, y / 100], [x2 / 100, y / 100]]
    if st.toggle("Đảo chiều vào / ra"):
        line = [line[1], line[0]]

    st.header("3. Tuỳ chọn")
    with st.expander("Mô hình & tốc độ"):
        conf = st.slider("Ngưỡng tin cậy", 0.1, 0.9, float(load_config()["model"]["conf"]), 0.05)
        imgsz = st.select_slider("Kích thước ảnh suy luận", [320, 480, 640, 960, 1280], value=640,
                                 help="Nhỏ hơn → nhanh hơn. Tăng lên 960–1280 cho cảnh đông, xe máy nhỏ/ở xa")
        tracker = st.selectbox("Thuật toán tracking", list(TRACKERS), format_func=TRACKERS.get)
    with st.expander("Thời gian & lưu trữ"):
        now = datetime.now()
        d = st.date_input("Ngày quay video", now.date())
        t = st.time_input("Giờ bắt đầu", dtime(now.hour, now.minute))
        st.caption("Thời điểm xe qua vạch = giờ bắt đầu + vị trí trong video (webcam/RTSP dùng giờ thực).")
        save_video = st.checkbox("Lưu video kết quả (.mp4)", value=True)

if not source:
    st.info("Chọn hoặc tải lên một video ở thanh bên trái để bắt đầu.", icon=":material/arrow_back:")
    st.stop()

# ---------------- Bo cuc chinh ----------------
b1, b2, _ = st.columns([1.3, 0.8, 4])
start = b1.button("Bắt đầu phân tích", type="primary", icon=":material/play_arrow:", width="stretch")
b2.button("Dừng", icon=":material/stop:", width="stretch")

left, right = st.columns([2, 1], gap="large")
with left:
    frame_ph = st.empty()
    progress_ph = st.empty()
    after_ph = st.empty()
with right:
    stats_ph = st.empty()
    feed_ph = st.empty()


def render_stats(by_class: dict, n_in: int, n_out: int, fps: float | None, title: str = "Số liệu trực tiếp"):
    with stats_ph.container():
        st.markdown(f"**{title}**")
        kpi("Tổng phương tiện", sum(by_class.values()), "#0F172A")
        for col, (cls, label) in zip(st.columns(3), CLASS_LABELS_VI.items()):
            with col:
                kpi(label, by_class.get(cls, 0), CLASS_COLORS_HEX[cls], small=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Vào ↓", n_in, "#6366F1", small=True)
        with c2:
            kpi("Ra ↑", n_out, "#EC4899", small=True)
        with c3:
            if fps is None:
                kpi("FPS", "–", "#94A3B8", small=True)
            else:
                ok = fps >= 15
                kpi("FPS", f"{fps:.1f}", "#22C55E" if ok else "#EF4444", "đạt ≥ 15" if ok else "< 15", small=True)


def render_feed(feed):
    rows = "".join(
        f"<div>{html.escape(ts)} &nbsp; {pill(cls)} &nbsp;#{tid} &nbsp;{'↓ vào' if d == 'in' else '↑ ra'}</div>"
        for ts, cls, tid, d in feed
    ) or "<div style='color:#94A3B8'>Chưa có xe nào qua vạch</div>"
    feed_ph.markdown(f"<br>**Xe vừa qua vạch**<div class='feed'>{rows}</div>", unsafe_allow_html=True)


def render_downloads(last):
    with after_ph.container():
        c1, c2, _ = st.columns([1, 1, 0.6])
        if c1.button("Xem trong Dashboard", icon=":material/monitoring:", width="stretch"):
            st.switch_page("views/dashboard.py")
        out = last.get("output")
        if out and Path(out).exists():
            c2.download_button("Tải video kết quả", Path(out).read_bytes(), file_name=Path(out).name,
                               mime="video/mp4", icon=":material/download:", width="stretch")


if not start:
    try:
        frame_ph.image(to_display(preview_with_line(first_frame(source), line)), width="stretch",
                       caption="Xem trước vạch đếm — mũi tên chỉ chiều VÀO. Chỉnh vạch ở thanh bên trái.")
    except RuntimeError as e:
        frame_ph.error(str(e))

    last = st.session_state.get("last_run")
    if last:
        render_stats(last["by_class"], last["in"], last["out"], last["fps"],
                     f"Kết quả phiên gần nhất (#{last['session_id']})")
        render_feed(last["feed"])
        render_downloads(last)
    else:
        render_stats({}, 0, 0, None)
        render_feed([])
    st.stop()

# ---------------- Chay pipeline ----------------
pipe = TrafficPipeline(model=get_model(str(default_weights())), conf=conf, imgsz=imgsz, tracker=tracker, db=get_db())
output = None
if save_video:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = str(OUTPUT_DIR / f"ketqua_{datetime.now():%Y%m%d_%H%M%S}.mp4")

feed = deque(maxlen=10)
render_stats({}, 0, 0, None)
render_feed(feed)
res = None
try:
    for res in pipe.run(source, line, start_time=datetime.combine(d, t), output_path=output):
        for e in res.new_events:
            feed.appendleft((mmss(e.frame_idx / res.video_fps), e.class_name, e.track_id, e.direction))
        # luu lien tuc de khi bam Dung van xem lai duoc ket qua
        st.session_state["last_run"] = {
            "session_id": res.session_id, "by_class": dict(res.by_class), "in": res.counts_in.total(),
            "out": res.counts_out.total(), "fps": res.fps, "output": output, "feed": list(feed),
        }
        frame_ph.image(to_display(res.frame), width="stretch")
        if res.total_frames:
            progress_ph.progress(res.progress, text=f"Khung hình {res.frame_idx}/{res.total_frames} · "
                                                    f"{mmss(res.video_seconds)}")
        if res.new_events or res.frame_idx % 5 == 1:
            render_stats(res.by_class, res.counts_in.total(), res.counts_out.total(), res.fps)
        if res.new_events:
            render_feed(feed)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if res is None:
    st.warning("Nguồn video không có khung hình nào.")
    st.stop()

render_stats(res.by_class, res.counts_in.total(), res.counts_out.total(), res.fps, "Kết quả")
progress_ph.success(f"Hoàn tất! Đã đếm **{res.total}** phương tiện trong {res.frame_idx} khung hình "
                    f"(phiên #{res.session_id}).", icon=":material/task_alt:")
render_downloads(st.session_state["last_run"])
