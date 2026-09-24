import streamlit as st

from traffic.classes import CLASS_NAMES
from traffic.pipeline import default_weights
from ui import class_kpis, get_db, hero

hero(
    "Hệ thống nhận diện, đếm và phân loại phương tiện giao thông",
    "Ứng dụng thị giác máy tính (YOLO26 + ByteTrack) biến video camera sẵn có thành số liệu lưu lượng "
    "theo thời gian và theo loại xe.",
)

db = get_db()
summary = db.summary()
by_class = {c: int(summary[summary.class_name == c]["count"].sum()) for c in CLASS_NAMES}
sessions = db.sessions()
class_kpis(by_class, sum(by_class.values()), {"total": f"{len(sessions)} phiên phân tích đã lưu"})

st.write("")
st.subheader("Luồng xử lý")
steps = [
    ("Video / Camera", "File video, webcam hoặc luồng RTSP"),
    ("Phát hiện", "YOLO26 fine-tune: xe con, xe tải, xe máy"),
    ("Theo dõi", "ByteTrack gán ID duy nhất cho mỗi xe"),
    ("Đếm & phân loại", "Đếm khi xe cắt vạch, mỗi ID một lần"),
    ("Lưu trữ", "SQLite: loại xe, chiều, thời điểm"),
    ("API & Dashboard", "FastAPI + biểu đồ trực quan"),
]
cols = st.columns(len(steps))
for i, (col, (t, d)) in enumerate(zip(cols, steps), start=1):
    col.markdown(f'<div class="step"><div class="n">{i}</div><div class="t">{t}</div><div class="d">{d}</div></div>',
                 unsafe_allow_html=True)

st.write("")
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Bắt đầu nhanh")
    st.markdown(
        """
1. Mở trang **Phân tích video**, chọn video mẫu hoặc tải video của bạn lên.
2. Kéo thanh trượt để đặt **vạch đếm** ngang qua làn xe — xem trước ngay trên ảnh.
3. Bấm **Bắt đầu phân tích** và theo dõi số liệu cập nhật theo thời gian thực.
4. Xem lại toàn bộ kết quả ở **Dashboard thống kê**, lọc theo phiên / thời gian, xuất CSV.
"""
    )
    c1, c2, _ = st.columns([1, 1, 1])
    if c1.button("Phân tích video ngay", type="primary", icon=":material/play_circle:", width="stretch"):
        st.switch_page("views/analyze.py")
    if c2.button("Xem dashboard", icon=":material/monitoring:", width="stretch"):
        st.switch_page("views/dashboard.py")

with right:
    st.subheader("Trạng thái hệ thống")
    w = default_weights()
    trained = "vehicle_yolo26" in str(w)
    with st.container(border=True):
        st.markdown(f"**Mô hình:** `{w.name}`")
        if trained:
            st.success("Đang dùng mô hình YOLO26 đã fine-tune cho 3 lớp xe.", icon=":material/check_circle:")
        else:
            st.warning("Chưa có mô hình fine-tune — đang dùng YOLO26n pretrained COCO "
                       "(bus gộp vào xe tải). Chạy `python scripts/train.py` để huấn luyện.",
                       icon=":material/warning:")
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        st.markdown(f"**Thiết bị suy luận:** {dev}")
        st.markdown(f"**CSDL:** `{db.path}`")
