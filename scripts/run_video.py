"""Chay pipeline detect -> track -> dem tren mot video (khong can giao dien).

Vi du:
    python scripts/run_video.py Data/videos/demo_traffic_sample.mp4 --output Data/outputs/demo.mp4 --show
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from traffic.classes import CLASS_LABELS_VI  # noqa: E402
from traffic.config import load_config, resolve  # noqa: E402
from traffic.pipeline import TrafficPipeline, load_model  # noqa: E402
from traffic.storage import TrafficDB  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="Duong dan video, chi so webcam (0) hoac URL RTSP")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--line", type=float, nargs=4, metavar=("AX", "AY", "BX", "BY"),
                    help="Vach dem theo toa do chuan hoa 0..1")
    ap.add_argument("--output", help="Luu video ket qua (.mp4)")
    ap.add_argument("--start-time", help="Thoi diem bat dau video, dang 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--no-db", action="store_true", help="Khong ghi vao CSDL")
    ap.add_argument("--show", action="store_true", help="Hien cua so xem truc tiep (nhan q de thoat)")
    args = ap.parse_args()

    line = [[args.line[0], args.line[1]], [args.line[2], args.line[3]]] if args.line else cfg["counting"]["line"]
    start = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M") if args.start_time else None
    db = None if args.no_db else TrafficDB(resolve(cfg["storage"]["db_path"]))
    pipe = TrafficPipeline(model=load_model(args.weights), db=db)

    res = None
    for res in pipe.run(args.source, line, start_time=start, output_path=args.output):
        if args.show:
            cv2.imshow("Traffic Flow Analytics", res.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif res.frame_idx % 100 == 0:
            print(f"  frame {res.frame_idx}/{res.total_frames}  fps={res.fps:.1f}  total={res.total}")
    cv2.destroyAllWindows()

    if res:
        print(f"\nPhien #{res.session_id}: {res.frame_idx} khung hinh, FPS ~{res.fps:.1f}")
        print(f"Tong: {res.total}  (vao {res.counts_in.total()} / ra {res.counts_out.total()})")
        for cls, label in CLASS_LABELS_VI.items():
            print(f"  {label:<8} {res.by_class.get(cls, 0)}")


if __name__ == "__main__":
    main()
