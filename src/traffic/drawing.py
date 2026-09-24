import cv2
import numpy as np

from .classes import CLASS_COLORS_BGR, CLASS_NAMES

FONT = cv2.FONT_HERSHEY_SIMPLEX
SHORT = {"car": "CAR", "truck": "TRUCK", "motorbike": "MOTO"}


def ui_scale(width: int) -> float:
    """He so phong to net ve theo do rong khung hinh (1.0 voi video 640px)."""
    return max(width / 640, 1.0)


def _i(v: float) -> int:
    return max(int(round(v)), 1)


def draw_line(frame, a, b, flash: bool = False, s: float = 1.0):
    color = (80, 220, 255) if flash else (60, 60, 255)
    cv2.line(frame, a, b, (0, 0, 0), _i(6 * s), cv2.LINE_AA)
    cv2.line(frame, a, b, color, _i(3 * s), cv2.LINE_AA)
    for p in (a, b):
        cv2.circle(frame, p, _i(7 * s), color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, _i(7 * s), (255, 255, 255), _i(2 * s), cv2.LINE_AA)


def draw_box(frame, box, cls, track_id, conf, counted: bool, s: float = 1.0):
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS_BGR[cls]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, _i((3 if counted else 2) * s), cv2.LINE_AA)
    label = f"#{track_id} {SHORT[cls]} {conf:.2f}" if track_id is not None else f"{SHORT[cls]} {conf:.2f}"
    fs, ft, pad = 0.45 * s, _i(s), _i(3 * s)
    (tw, th), _ = cv2.getTextSize(label, FONT, fs, ft)
    ty = max(y1, th + 2 * pad)
    cv2.rectangle(frame, (x1, ty - th - 2 * pad), (x1 + tw + 2 * pad, ty), color, -1)
    cv2.putText(frame, label, (x1 + pad, ty - pad - 1), FONT, fs, (255, 255, 255), ft, cv2.LINE_AA)
    if counted:
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), _i(4 * s), (255, 255, 255), -1, cv2.LINE_AA)


def draw_hud(frame, counts_by_class, total, fps, s: float = 1.0):
    """Bang so lieu ban trong suot o goc tren trai."""
    m, row = _i(10 * s), 24 * s
    h, w = _i(34 * s + row * (len(CLASS_NAMES) + 1)), _i(190 * s)
    overlay = frame.copy()
    cv2.rectangle(overlay, (m, m), (m + w, m + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    x = _i(20 * s)
    cv2.putText(frame, f"TOTAL  {total}", (x, _i(36 * s)), FONT, 0.65 * s, (255, 255, 255), _i(2 * s), cv2.LINE_AA)
    for i, cls in enumerate(CLASS_NAMES):
        y = _i(62 * s + row * i)
        cv2.rectangle(frame, (x, y - _i(11 * s)), (x + _i(12 * s), y + _i(s)), CLASS_COLORS_BGR[cls], -1)
        cv2.putText(frame, f"{SHORT[cls]:<6} {counts_by_class.get(cls, 0)}", (_i(40 * s), y), FONT, 0.5 * s,
                    (235, 235, 235), _i(s), cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:5.1f}", (x, _i(62 * s + row * len(CLASS_NAMES))), FONT, 0.5 * s, (150, 230, 150),
                _i(s), cv2.LINE_AA)


def preview_with_line(frame: np.ndarray, line_norm) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    s = ui_scale(w)
    a = (int(line_norm[0][0] * w), int(line_norm[0][1] * h))
    b = (int(line_norm[1][0] * w), int(line_norm[1][1] * h))
    draw_line(img, a, b, s=s)
    # mui ten chi chieu "vao" (phia phai cua vector AB)
    mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = max((dx * dx + dy * dy) ** 0.5, 1)
    nx, ny = -dy / n, dx / n
    tip = (int(mx + nx * 40 * s), int(my + ny * 40 * s))
    cv2.arrowedLine(img, (mx, my), tip, (80, 220, 255), _i(3 * s), cv2.LINE_AA, tipLength=0.35)
    cv2.putText(img, "IN", (tip[0] + _i(6 * s), tip[1] + _i(6 * s)), FONT, 0.6 * s, (80, 220, 255), _i(2 * s),
                cv2.LINE_AA)
    return img
