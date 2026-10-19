import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

from ui import finalize_run, inject_css  # noqa: E402

st.set_page_config(page_title="Traffic Flow Analytics", page_icon=":material/traffic:", layout="wide")
inject_css()

pages = [
    st.Page("views/overview.py", title="Tổng quan", icon=":material/home:", default=True),
    st.Page("views/analyze.py", title="Phân tích video", icon=":material/videocam:"),
    st.Page("views/dashboard.py", title="Dashboard thống kê", icon=":material/monitoring:"),
    st.Page("views/model.py", title="Đánh giá mô hình", icon=":material/model_training:"),
]

with st.sidebar:
    st.markdown("### Traffic Flow Analytics")
    st.caption("Nhận diện · Theo dõi · Đếm · Phân loại phương tiện bằng YOLO26 + ByteTrack")

page = st.navigation(pages)
# Phien phan tich dang tam dung chi duoc luu khi roi khoi trang Phan tich video
if page.url_path != "analyze":
    finalize_run()
page.run()
