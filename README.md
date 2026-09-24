# YOLO Traffic Flow Analytics

Hệ thống **nhận diện → theo dõi → đếm → phân loại** phương tiện giao thông (xe con, xe tải, xe máy) từ video/camera,
lưu kết quả vào CSDL và trực quan hoá trên dashboard.

> Đề tài thực tập — Trung tâm Thống kê và Nền tảng số, Cục Chuyển đổi số, Bộ Nông nghiệp và Môi trường.
> Sinh viên: Bùi Phương Duy (DCCTKH67A) · GVHD: Ngô Hùng Long · Hướng dẫn tại đơn vị: Mai Đức Anh
>
> Tài liệu tổng hợp toàn bộ dự án theo 10 tuần (kèm hình minh hoạ & tài liệu tham khảo):
> [docs/BAO_CAO_DU_AN.md](docs/BAO_CAO_DU_AN.md)

![Kiến trúc](docs/images/01_kien_truc_he_thong.png)

## 1. Kiến trúc

```
Video / Webcam / RTSP
      │
      ▼
[1] Phát hiện ── YOLO26n fine-tune (3 lớp: car, truck, motorbike)
      ▼
[2] Theo dõi ─── ByteTrack / BoT-SORT (gán ID cho từng xe)
      ▼
[3] Đếm ──────── xe cắt vạch đếm → +1 theo loại & chiều (mỗi ID chỉ đếm 1 lần,
      │           loại xe = lớp được "bỏ phiếu" nhiều nhất qua các khung hình)
      ▼
[4] Lưu trữ ──── SQLite: sessions + vehicle_events (loại xe, chiều, thời điểm)
      ▼
[5] API ──────── FastAPI  (/stats/summary, /stats/timeseries, /events, /sessions)
[6] Dashboard ── Streamlit (phân tích video trực tiếp + thống kê + đánh giá mô hình)
```

## 2. Cấu trúc thư mục

```
├── app/                      # Giao diện demo Streamlit
│   ├── streamlit_app.py      #   điểm vào, điều hướng các trang
│   ├── ui.py                 #   CSS + thành phần giao diện dùng chung
│   └── views/                #   overview · analyze · dashboard · model
├── src/traffic/              # Thư viện lõi
│   ├── pipeline.py           #   video → detect → track → count → DB
│   ├── counter.py            #   logic đếm qua vạch (thuần Python, có unit test)
│   ├── storage.py            #   SQLite + truy vấn thống kê
│   ├── api.py                #   FastAPI
│   ├── drawing.py            #   vẽ box, vạch đếm, bảng số liệu lên khung hình
│   ├── classes.py            #   tên lớp, nhãn tiếng Việt, màu, ánh xạ COCO
│   └── config.py
├── scripts/
│   ├── prepare_dataset.py    # chuẩn hoá nhãn 4 lớp → 3 lớp + kiểm tra/thống kê bộ dữ liệu
│   ├── download_videos.py    # tải lại các video giao thông Việt Nam để thử nghiệm
│   ├── train.py              # huấn luyện YOLO26
│   ├── evaluate.py           # đánh giá trên tập test + đo FPS pipeline
│   ├── benchmark.py          # so sánh tốc độ/độ chính xác theo imgsz, PyTorch vs ONNX
│   ├── make_figures.py       # sinh hình minh hoạ cho báo cáo (docs/images)
│   └── run_video.py          # chạy pipeline từ dòng lệnh
├── configs/
│   ├── config.yaml           # model, ngưỡng, tracker, vạch đếm mặc định, đường dẫn DB
│   └── dataset.yaml          # cấu hình bộ dữ liệu cho Ultralytics
├── models/vehicle_yolo26/    # kết quả huấn luyện (weights/best.pt, results.csv, biểu đồ)
├── tests/                    # pytest (unit + tích hợp)
├── docs/                     # tài liệu tổng hợp dự án + hình minh hoạ
├── Data/                     # bộ dữ liệu + video + CSDL  (KHÔNG đưa lên GitHub, xem mục 3)
└── run_demo.bat              # khởi chạy nhanh demo trên Windows
```

## 3. Cài đặt

Yêu cầu: Python 3.10+ (đã thử với 3.11), khuyến nghị GPU NVIDIA.

```bash
python -m venv venv
venv\Scripts\activate                 # Linux/macOS: source venv/bin/activate

# PyTorch có CUDA (bỏ --index-url nếu chỉ dùng CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Chuẩn bị dữ liệu

Thư mục `Data/` không được đưa lên GitHub (dung lượng lớn), cần tự chuẩn bị sau khi clone:

1. Tải bộ dữ liệu [Vietnamese vehicle v3](https://universe.roboflow.com/car-classification/vietnamese-vehicle/dataset/3)
   (Roboflow, định dạng YOLO), giải nén thành `Data/Vietnamese vehicle.v3-2023-02-01-5-31pm.yolo26/`.
2. Chuẩn hoá nhãn về 3 lớp và kiểm tra:
   ```bash
   python scripts/prepare_dataset.py --remap
   ```
3. Tải video thử nghiệm: `python scripts/download_videos.py`

Mô hình đã huấn luyện `models/vehicle_yolo26/weights/best.pt` có sẵn trong repo nên có thể chạy demo ngay
mà không cần bước 1–2.

## 4. Chạy demo

**Cách nhanh (Windows):** nhấp đúp `run_demo.bat` → trình duyệt mở `http://localhost:8501`.

**Hoặc chạy thủ công:**

```bash
streamlit run app/streamlit_app.py                        # giao diện demo
uvicorn traffic.api:app --app-dir src --port 8000         # API, tài liệu tại http://localhost:8000/docs
```

### Các trang trong giao diện

| Trang | Chức năng |
| --- | --- |
| **Tổng quan** | Giới thiệu hệ thống, luồng xử lý, tổng số xe đã đếm, trạng thái mô hình/GPU |
| **Phân tích video** | Chọn video mẫu / tải video lên / webcam / RTSP → kéo thanh trượt đặt vạch đếm (xem trước ngay) → bấm **Bắt đầu phân tích**: video có box + ID chạy trực tiếp, số xe theo loại & chiều, FPS, danh sách xe vừa qua vạch. Tải về video kết quả. |
| **Dashboard thống kê** | Lọc theo phiên / khoảng ngày / mức gom nhóm (10 giây → ngày). KPI, biểu đồ lưu lượng theo thời gian, cơ cấu loại xe, theo chiều, tích luỹ; bảng chi tiết & **xuất CSV**; quản lý/xoá phiên |
| **Đánh giá mô hình** | mAP, Precision, Recall (tổng & từng lớp), FPS pipeline, đường cong huấn luyện, ma trận nhầm lẫn, PR curve |

## 5. Huấn luyện & đánh giá

```bash
python scripts/train.py                 # mặc định: yolo26n.pt, 80 epoch, imgsz 640, batch 16
python scripts/evaluate.py              # đánh giá tập test + FPS → models/vehicle_yolo26/eval_test.json
python scripts/benchmark.py             # tốc độ/độ chính xác theo imgsz, PyTorch vs ONNX → benchmark.json
python scripts/make_figures.py          # sinh lại hình minh hoạ trong docs/images
```

Bộ dữ liệu: *Vietnamese vehicle* (Roboflow, CC BY 4.0) — 1235 train / 156 valid / 156 test ảnh, đã chuẩn hoá về
3 lớp `car`, `truck`, `motorbike` (nhãn gốc được sao lưu ở `Data/processed_labels_backup`).

Nếu chưa có `models/vehicle_yolo26/weights/best.pt`, hệ thống tự dùng `yolo26n.pt` pretrained COCO
(`bus` được gộp vào xe tải, `motorcycle` → xe máy).

### Video thử nghiệm (`Data/videos/`)

Các video `vn_*.mp4` tải từ [Pexels](https://www.pexels.com/license/) (miễn phí, được phép sử dụng), bản 720p:

| Tệp | Nội dung | Nguồn |
| --- | --- | --- |
| `vn_giao_thong_ban_ngay.mp4` | Giao lộ ban ngày, nhiều xe máy + ô tô | [Pexels 33383214](https://www.pexels.com/video/bustling-vietnam-traffic-on-sunny-day-33383214/) |
| `vn_nga_tu_xe_may_o_to.mp4` | Ngã tư, xe máy + ô tô | [Pexels 30247347](https://www.pexels.com/video/bustling-city-intersection-with-motorbikes-and-cars-30247347/) |
| `vn_hcm_nga_tu_ban_dem.mp4` | Ngã tư TP.HCM ban đêm | [Pexels 33486917](https://www.pexels.com/video/busy-night-intersection-in-ho-chi-minh-city-33486917/) |
| `vn_ket_xe_xe_may_o_to.mp4` | Kẹt xe, mật độ xe máy rất cao | [Pexels 39221451](https://www.pexels.com/video/dense-urban-traffic-jam-with-motorbikes-and-cars-39221451/) |
| `vn_hanoi_pho_xe_may_xe_buyt.mp4` | Đường phố, camera di chuyển theo dòng xe | [Pexels 33383212](https://www.pexels.com/video/bustling-hanoi-street-with-motorbikes-and-buses-33383212/) |
| `vn_cao_toc_flycam.mp4` | Quốc lộ quay từ flycam (góc trên cao) | [Pexels 32971521](https://www.pexels.com/video/aerial-view-of-traffic-on-busy-vietnamese-highway-32971521/) |

Hai video cuối nằm ngoài phạm vi đề tài (camera không cố định / góc nhìn trên cao), dùng để minh hoạ giới hạn của mô hình.

## 6. Chạy từ dòng lệnh

```bash
python scripts/run_video.py Data/videos/vn_giao_thong_ban_ngay.mp4 --output Data/outputs/demo.mp4 --show
python scripts/run_video.py 0 --line 0.1 0.5 0.9 0.5                 # webcam, vạch ngang giữa khung hình
python scripts/run_video.py video.mp4 --start-time "2026-09-01 07:00"  # gán thời điểm thực cho video
```

## 7. API

| Endpoint | Mô tả |
| --- | --- |
| `GET /health` | Kiểm tra trạng thái |
| `GET /classes` | Danh sách lớp |
| `GET /sessions` | Danh sách phiên phân tích + số xe |
| `DELETE /sessions/{id}` | Xoá một phiên |
| `GET /stats/summary?start=&end=&session_id=` | Tổng số xe theo loại và theo chiều |
| `GET /stats/timeseries?interval=minute\|hour\|day` | Chuỗi thời gian số xe theo loại |
| `GET /events?limit=100` | Danh sách từng xe đã đếm |

## 8. Kiểm thử

```bash
pytest
```

Gồm unit test cho logic đếm (đi qua vạch theo 2 chiều, không đếm trùng khi xe rung trên vạch, ngoài đoạn vạch,
bỏ phiếu lớp), test tích hợp SQLite + API và test end-to-end chạy pipeline thật trên video mẫu
(tự bỏ qua nếu thiếu video/weights).
