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
from traffic.pipeline import TrafficPipeline, default_weights, is_live_source, load_model, read_first_frame
from ui import finalize_run, get_db, kpi, pill, run_holder, set_run

SAMPLE_DIR = PROJECT_ROOT / "Data" / "videos"
UPLOAD_DIR = PROJECT_ROOT / "Data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "outputs"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
DISPLAY_MAX_W = 960
TRACKERS = {"bytetrack.yaml": "ByteTrack", "botsort.yaml": "BoT-SORT"}
EMPTY_HINT = {
    "Video mẫu": "Chọn một video mẫu ở thanh bên trái để bắt đầu.",
    "Tải video lên": "Tải lên một video ở thanh bên trái để bắt đầu.",
    "Camera IP (RTSP)": "Nhập địa chỉ luồng RTSP ở thanh bên trái để bắt đầu.",
}


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
st.caption("Chọn nguồn video, đặt vạch đếm rồi bấm **Bắt đầu phân tích**. Có thể **Dừng** rồi **Tiếp tục**; phiên được "
           "lưu vào CSDL khi video kết thúc, khi đổi video hoặc khi rời trang này.")

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
        save_video = st.checkbox("Lưu video kết quả (.mp4)", value=False, help="Tệp kết quả được lưu trong Data/outputs. Chỉ dùng khi video đầu vào là tệp hoặc webcam.")

# ---------------- Phien phan tich ----------------
# st.session_state["run"] giu phien qua cac lan chay lai cua trang: generator cua pipeline + anh chup ket qua gan nhat.
# status: running | paused | done | error. Dung -> paused: generator van song, CSDL chua dong phien (chua ghi gio ket
# thuc). Phien chi duoc dong khi video ket thuc, doi nguon, bam Bat dau lai, hoac roi trang (xem streamlit_app.py).
run = st.session_state.get("run")
if run and run["gen"] is not None and run["source"] != source:
    finalize_run()

if not source:
    st.info(EMPTY_HINT.get(mode, "Chọn nguồn video ở thanh bên trái để bắt đầu."), icon=":material/arrow_back:")
    st.stop()


def request_start():
    finalize_run()
    st.session_state["start_requested"] = True


def request_pause():
    r = st.session_state["run"]
    if r["live"]:  # webcam/RTSP chay lien tuc, khong tam dung duoc nen Dung = ket thuc phien
        finalize_run()
    else:
        r["status"] = "paused"


def request_resume():
    st.session_state["run"]["status"] = "running"


if st.session_state.pop("start_requested", False):
    pipe = TrafficPipeline(model=get_model(str(default_weights())), conf=conf, imgsz=imgsz, tracker=tracker, db=get_db())
    output = None
    if save_video:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output = str(OUTPUT_DIR / f"ketqua_{datetime.now():%Y%m%d_%H%M%S}.mp4")
    run = {
        "gen": pipe.run(source, line, start_time=datetime.combine(d, t), output_path=output),
        "source": source, "live": is_live_source(source), "status": "running", "ended": None, "error": None,
        "output": output, "feed": deque(maxlen=10), "snap": None,
    }
    set_run(run)

status = run["status"] if run and run["source"] == source else "idle"

# ---------------- Bo cuc chinh ----------------
b1, b2, b3, _ = st.columns([1.3, 0.8, 1, 3.4])
b1.button("Bắt đầu lại" if status == "paused" else "Bắt đầu phân tích", type="primary", icon=":material/play_arrow:",
          width="stretch", disabled=status == "running", on_click=request_start, key="btn_start")
b2.button("Dừng", icon=":material/stop:", width="stretch", disabled=status != "running", on_click=request_pause,
          key="btn_pause")
b3.button("Tiếp tục", icon=":material/resume:", width="stretch", disabled=status != "paused", on_click=request_resume,
          key="btn_resume",
          help=("Webcam/RTSP không tạm dừng được vì luồng vẫn chạy trong lúc dừng; bấm Dừng sẽ kết thúc phiên."
                if is_live_source(source) else None))

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


def render_snapshot(snap, title):
    render_stats(snap["by_class"], snap["in"], snap["out"], snap["fps"], title)
    render_feed(snap["feed"])


def render_downloads(snap):
    with after_ph.container():
        c1, c2, _ = st.columns([1, 1, 0.6])
        if c1.button("Xem trong Dashboard", icon=":material/monitoring:", width="stretch"):
            st.switch_page("views/dashboard.py")
        out = snap.get("output")
        if out and Path(out).exists():
            c2.download_button("Tải video kết quả", Path(out).read_bytes(), file_name=Path(out).name,
                               mime="video/mp4", icon=":material/download:", width="stretch")


if status == "running":
    # ---------------- Dang chay: doc tiep khung tu generator ----------------
    feed, error = run["feed"], None
    if run["snap"]:
        render_snapshot(run["snap"], "Số liệu trực tiếp")
    else:
        render_stats({}, 0, 0, None)
        render_feed(feed)
    try:
        for res in run["gen"]:
            for e in res.new_events:
                feed.appendleft((mmss(e.frame_idx / res.video_fps), e.class_name, e.track_id, e.direction))
            disp = to_display(res.frame)
            # Cap nhat anh chup TRUOC cac lenh st.*: bam Dung ngat vong lap tai lenh st.* ke tiep
            run["snap"] = {
                "session_id": res.session_id, "by_class": dict(res.by_class), "in": res.counts_in.total(),
                "out": res.counts_out.total(), "fps": res.fps, "output": run["output"], "feed": list(feed),
                "frame": disp, "frame_idx": res.frame_idx, "total_frames": res.total_frames,
                "seconds": res.video_seconds, "total": res.total,
            }
            frame_ph.image(disp, width="stretch")
            if res.total_frames:
                progress_ph.progress(res.progress, text=f"Khung hình {res.frame_idx}/{res.total_frames} · "
                                                        f"{mmss(res.video_seconds)}")
            if res.new_events or res.frame_idx % 5 == 1:
                render_stats(res.by_class, res.counts_in.total(), res.counts_out.total(), res.fps)
            if res.new_events:
                render_feed(feed)
    except RuntimeError as e:
        error = str(e)
    except BaseException:
        if run_holder().get("released"):  # tab da dong giua luc dang chay: dong phien roi moi thoat
            run["gen"].close()
        raise
    run["gen"] = None  # generator da het (video ket thuc hoac loi): pipeline da dong phien trong khoi finally
    if error:
        run["status"], run["error"] = "error", error
    elif run["snap"] is None:
        run["status"], run["error"] = "error", "Nguồn video không có khung hình nào."
    else:
        run["status"], run["ended"] = "done", "video"
    st.rerun()  # ve lai de cac nut doi trang thai

elif status == "paused":
    # ---------------- Tam dung: hien khung cuoi, so lieu, chua dong phien ----------------
    snap = run["snap"]
    frame_ph.image(snap["frame"], width="stretch")
    if snap["total_frames"]:
        progress_ph.progress(min(snap["frame_idx"] / snap["total_frames"], 1.0),
                             text=f"Khung hình {snap['frame_idx']}/{snap['total_frames']} · {mmss(snap['seconds'])}")
    render_snapshot(snap, f"Đã tạm dừng · phiên #{snap['session_id']}")
    after_ph.info("Bấm **Tiếp tục** để chạy tiếp từ khung hình này. Phiên được lưu khi video kết thúc, khi đổi video "
                  "hoặc khi rời trang này.", icon=":material/pause_circle:")

else:
    # ---------------- Chua chay / da xong / loi ----------------
    if status == "error":
        frame_ph.error(run["error"], icon=":material/error:")
        render_stats({}, 0, 0, None)
        render_feed([])
    else:
        try:
            frame_ph.image(to_display(preview_with_line(first_frame(source), line)), width="stretch",
                           caption="Xem trước vạch đếm — mũi tên chỉ chiều VÀO. Chỉnh vạch ở thanh bên trái.")
        except RuntimeError as e:
            frame_ph.error(str(e), icon=":material/error:")
        snap = run["snap"] if run else None
        if snap:
            if run["ended"] == "video":
                msg = (f"Hoàn tất! Đã đếm **{snap['total']}** phương tiện trong {snap['frame_idx']} khung hình "
                       f"(phiên #{snap['session_id']}).")
            else:
                msg = (f"Đã dừng và lưu phiên #{snap['session_id']}: {snap['total']} phương tiện trong "
                       f"{snap['frame_idx']} khung hình.")
            progress_ph.success(msg, icon=":material/task_alt:")
            render_snapshot(snap, f"Kết quả phiên gần nhất (#{snap['session_id']})")
            render_downloads(snap)
        else:
            render_stats({}, 0, 0, None)
            render_feed([])
