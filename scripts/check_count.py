"""Doi chieu so xe he thong dem voi so DEM TAY tren mot video -> models/vehicle_yolo26/count_check.json.

Chay pipeline day du (phat hien + ByteTrack + vach dem) voi vai gia tri conf, so voi so dem tay do nguoi thuc hien nhap.
Sai so tinh theo tung lop va tong. Luu y: day la so SO SANH SO LUONG, khong ghep tung xe mot voi nhau, nen hai loi
nguoc chieu (dem thua + dem thieu) co the triet tieu nhau trong tong; vi vay in them tong sai so tuyet doi theo lop.

Vi du:
  python scripts/check_count.py --hand car=8 truck=1 motorbike=21
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from traffic.classes import CLASS_NAMES  # noqa: E402
from traffic.config import load_config  # noqa: E402
from traffic.pipeline import TrafficPipeline  # noqa: E402


def count(video, conf, line):
    pipe = TrafficPipeline(conf=conf)
    last = None
    for r in pipe.run(video, line):
        last = r
    return {c: int(last.by_class.get(c, 0)) for c in CLASS_NAMES}


def errors(system, hand):
    per = {}
    for c in CLASS_NAMES:
        diff = system[c] - hand[c]
        per[c] = {"he_thong": system[c], "dem_tay": hand[c], "lech": diff,
                  "sai_so_tuyet_doi": abs(diff), "sai_so_phan_tram": (abs(diff) / hand[c] * 100) if hand[c] else None}
    ts, th = sum(system.values()), sum(hand.values())
    return {"theo_lop": per,
            "tong": {"he_thong": ts, "dem_tay": th, "lech": ts - th, "sai_so_phan_tram": abs(ts - th) / th * 100},
            "tong_sai_so_tuyet_doi_theo_lop": sum(v["sai_so_tuyet_doi"] for v in per.values())}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="Data/videos/vn_giao_thong_ban_ngay.mp4")
    ap.add_argument("--hand", nargs="+", required=True, metavar="lop=so", help="vd: car=8 truck=1 motorbike=21")
    ap.add_argument("--conf", nargs="+", type=float, default=[0.10, 0.20, 0.25, cfg["model"]["conf"]])
    args = ap.parse_args()

    hand = {c: 0 for c in CLASS_NAMES}
    for kv in args.hand:
        k, v = kv.split("=")
        hand[k] = int(v)
    out = {"video": Path(args.video).name, "vach": cfg["counting"]["line"], "dem_tay": hand,
           "conf_mac_dinh": cfg["model"]["conf"], "ket_qua": {}}
    for conf in sorted(set(args.conf)):
        sysc = count(args.video, conf, cfg["counting"]["line"])
        out["ket_qua"][f"{conf:.2f}"] = errors(sysc, hand)
        e = out["ket_qua"][f"{conf:.2f}"]
        print(f"conf {conf:.2f}: he thong {sysc} | dem tay {hand} | lech tong {e['tong']['lech']:+d} "
              f"({e['tong']['sai_so_phan_tram']:.1f}%) | tong |lech| theo lop {e['tong_sai_so_tuyet_doi_theo_lop']}", flush=True)
    path = ROOT / "models" / "vehicle_yolo26" / "count_check.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ghi", path)


if __name__ == "__main__":
    main()
