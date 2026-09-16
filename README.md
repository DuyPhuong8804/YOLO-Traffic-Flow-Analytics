# YOLO Traffic Flow Analytics

Hệ thống nhận diện, theo dõi, đếm và phân loại phương tiện giao thông (car/truck/motorbike) từ video camera giám sát cố định, phục vụ thống kê lưu lượng giao thông theo thời gian.

Đề tài thực tập — Trung tâm thống kê và nền tảng số, Cục Chuyển đổi số, Bộ Nông nghiệp và Môi trường.

## 1. Kiến trúc tổng thể

```
Video (file) → Detect (YOLO26) → Track (ByteTrack) → Count (đường kẻ ảo)
                                                            │
                                                            ▼
                                      Storage (SQLite, batch insert)
                                                            │
                                                            ▼
                                          Dashboard (Streamlit, đọc thẳng SQLite)
```

## 2. Cấu trúc thư mục

```
├── Data/                        # dataset + video mẫu (không commit weights/db)
│   ├── Vietnamese vehicle.v3.../ # dataset gốc (Roboflow), đã remap về 3 lớp
│   ├── videos/                   # video test (demo_traffic_sample.mp4)
│   └── class_inspection/         # ảnh minh hoạ xác minh class mapping
├── models/                       # weights sau khi train (models/vehicle_yolo26/weights/best.pt)
├── configs/
│   ├── train_config.yaml         # hyperparameter training
│   ├── counting_lines.yaml       # toạ độ đường kẻ ảo theo camera_id
│   └── app_config.yaml           # đường dẫn model/DB, tracker, ngưỡng conf/iou
├── src/
│   ├── common/                   # class mapping, config loader, vẽ overlay
│   ├── detection/                # train.py, evaluate.py, detector.py
│   ├── tracking/                 # wrapper ByteTrack qua Ultralytics
│   ├── counting/                 # thuật toán đếm qua đường kẻ ảo (lõi hệ thống)
│   ├── storage/                  # SQLite + batch insert + truy vấn thống kê (stats.py)
│   ├── dashboard/                # Streamlit app.py (đọc thẳng SQLite, không qua API)
│   └── pipeline.py               # orchestrator nối toàn bộ luồng xử lý
├── scripts/                      # prepare_dataset.py, inspect_classes.py, pick_line_coords.py
├── tests/                        # unit test cho logic đếm + storage (không cần GPU/model)
├── requirements.txt
└── README.md
```

## 3. Cài đặt

```powershell
# 1. Tạo/kích hoạt virtualenv (đã có sẵn venv/ trong repo)
venv\Scripts\activate

# 2. QUAN TRỌNG: cài torch có CUDA TRƯỚC (bản PyPI mặc định trên Windows là CPU-only)
#    Kiểm tra CUDA driver bằng `nvidia-smi`, chọn index-url phù hợp (vd cu121 cho CUDA 12.1+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Cài các thư viện còn lại
pip install -r requirements.txt
```

Kiểm tra GPU đã được nhận:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Chuẩn bị dữ liệu

Dataset gốc (`Data/Vietnamese vehicle.v3.../`) có 4 lớp: car(0), motorbike(1), truck(2), bus(3) — đã xác minh trực quan (xem `Data/class_inspection/`). Script sau sẽ **gộp bus → truck** và remap về 3 lớp cuối cùng `car(0)/truck(1)/motorbike(2)` (đã chạy sẵn, an toàn khi chạy lại — có backup và idempotent):

```powershell
python scripts/prepare_dataset.py
```

> Quy tắc gộp lớp: bus được xếp vào nhóm "phương tiện cỡ lớn" cùng truck do đặc điểm kích thước/tốc độ/mức chiếm dụng làn đường tương đồng — xem docstring trong `src/common/class_map.py`.

Nếu dùng dataset khác, chạy `scripts/inspect_classes.py` trước để xác minh lại thứ tự class id bằng ảnh minh hoạ, tránh gán nhầm lớp (đây là lỗi âm thầm rất dễ xảy ra vì `data.yaml` của Roboflow xuất ra không luôn ghi đúng tên lớp).

## 5. Training

```powershell
python -m src.detection.train --config configs/train_config.yaml
```

Cấu hình mặc định dùng `yolo26n.pt` (nano) — ưu tiên tốc độ để đạt real-time ≥15 FPS trên GPU VRAM thấp (vd RTX 3050 4GB). Nếu cần độ chính xác cao hơn và đủ VRAM, đổi `model: yolo26s.pt` trong `configs/train_config.yaml`.

Đánh giá mô hình (mục tiêu mAP@0.5 ≥ 0.7):
```powershell
python -m src.detection.evaluate --weights models/vehicle_yolo26/weights/best.pt --split test
```

## 6. Chạy thử pipeline (detect + track + count) trên video mẫu

Sau khi có `models/vehicle_yolo26/weights/best.pt`, cập nhật `model_path` trong `configs/app_config.yaml` (mặc định đã trỏ sẵn đến đường dẫn này), rồi chạy nhanh bằng Python:

```python
from src.pipeline import TrafficPipeline
import cv2

pipeline = TrafficPipeline(camera_id="demo_cam_01")
for result in pipeline.run():
    cv2.imshow("preview", result.frame)
    if cv2.waitKey(1) == 27:
        break
```

Video mẫu `Data/videos/demo_traffic_sample.mp4`: clip "Traffic Flow In The Highway" (tác giả Mike Bird, nguồn [Pexels](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/), Pexels License — miễn phí sử dụng). Quay từ **camera cố định trên cao**, cảnh xe chạy 2 chiều trên đường cao tốc nhiều làn, phù hợp để test cơ chế detect/track/count qua đường kẻ ảo theo cả 2 hướng. Dài 60s/1800 frame.

> Lưu ý: video này KHÔNG dùng để fine-tune (không có nhãn, và bối cảnh là đường cao tốc nước ngoài, khác đặc điểm giao thông Việt Nam) — chỉ dùng để kiểm thử cơ chế tracking/đếm. Xem mục 9 để biết cách bổ sung video thực tế Việt Nam phục vụ kiểm thử độ chính xác.

Để lấy toạ độ đường kẻ ảo cho 1 video/camera mới:
```powershell
python scripts/pick_line_coords.py --video "duong/dan/video.mp4"
```
rồi dán toạ độ in ra vào `configs/counting_lines.yaml`.

## 7. Chạy Dashboard

```powershell
streamlit run src/dashboard/app.py
```
- Tab **"Video trực tiếp"**: chạy pipeline ngay trong Streamlit, xem preview video có bounding box + đường kẻ ảo + số đếm.
- Tab **"Thống kê"**: đọc thẳng từ SQLite (`Data/traffic_counts.db`) — không cần chạy thêm service nào khác, tự làm mới mỗi 10s.

## 8. Kiểm thử độ chính xác đếm

Vì thuật toán đếm chỉ đúng khi cấu hình đường kẻ ảo phù hợp với góc quay thực tế, cần kiểm thử bằng **đếm tay đối chiếu (ground-truth thủ công)** trên video thực tế của đơn vị, không chỉ dựa vào mAP của detector:

1. **Chuẩn bị 3–5 clip ngắn (30–60 giây)** đại diện các kịch bản:
   - Lưu lượng thưa (dễ, baseline).
   - Lưu lượng đông, nhiều xe máy đi sát nhau (kiểm tra occlusion).
   - Điều kiện ánh sáng khác nhau (ban ngày / chiều tối / ngược sáng).
   - Góc camera khác nhau nếu triển khai nhiều điểm.
2. Xem từng clip bằng mắt, **đếm tay thủ công** số lượng xe mỗi loại đi qua theo từng hướng → ghi thành bảng ground-truth.
3. Chạy hệ thống trên cùng clip, so sánh số đếm hệ thống vs ground-truth:
   - Accuracy đếm = `1 - |predicted - actual| / actual` (tính riêng từng loại xe và từng hướng).
   - Theo dõi riêng 2 loại lỗi: **đếm trùng** (predicted > actual, thường do ID switch) và **đếm sót** (predicted < actual, thường do track bị mất khi occlusion quá lâu vượt `track_buffer`).
4. Nếu accuracy thấp ở kịch bản đông xe máy: tăng `track_buffer` trong `configs/app_config.yaml`, giảm `conf_threshold` một chút để bắt được xe bị che khuất một phần, hoặc cân nhắc `yolo26s` thay vì `yolo26n`.

Unit test cho riêng phần thuật toán đếm (không cần video/model thật, chạy được ngay):
```powershell
pytest tests/ -v
```

## 9. Hiệu năng

- Mục tiêu ≥15 FPS. `yolo26n` + GPU tầm trung (vd RTX 3050) thường đạt real-time ở `imgsz=640`.
- YOLO26 loại bỏ bước NMS hậu xử lý truyền thống (kiến trúc end-to-end one-to-one), giảm độ trễ so với các bản YOLO trước — quan trọng khi mật độ xe máy cao làm số lượng box cần NMS tăng mạnh.
- Nếu không đạt FPS mục tiêu: giảm `imgsz`, dùng `yolo26n` thay vì `s/m`, hoặc giảm tần suất chạy detect (detect mỗi N frame, dùng Kalman/track dự đoán các frame giữa).
