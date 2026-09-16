"""
Cong cu ho tro chon toa do 2 diem dau-cuoi cua duong ke ao tren 1 video/camera moi,
bang cach click chuot truc tiep len khung hinh dau tien cua video.

Sau khi chay, copy toa do in ra terminal vao configs/counting_lines.yaml
(truong point1/point2 cua duong ke tuong ung).

Chay:
    python scripts/pick_line_coords.py --video "Data/videos/demo_traffic_sample.mp4"
"""

import argparse

import cv2

points: list[tuple[int, int]] = []


def _on_click(event, x, y, flags, param) -> None:
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        print(f"Diem {len(points)}: ({x}, {y})")


def main(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Khong doc duoc frame dau tien tu {video_path}")

    window = "Click 2 diem de tao duong ke ao (ESC de thoat)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_click)

    while True:
        display = frame.copy()
        for p in points:
            cv2.circle(display, p, 5, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (0, 255, 255), 2)
        cv2.imshow(window, display)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break
        if len(points) == 2:
            print("\n>>> Dan vao configs/counting_lines.yaml:")
            print(f"point1: [{points[0][0]}, {points[0][1]}]")
            print(f"point2: [{points[1][0]}, {points[1][1]}]")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chon toa do duong ke ao bang click chuot")
    parser.add_argument("--video", required=True, help="Duong dan video de lay khung hinh mau")
    args = parser.parse_args()
    main(args.video)
