"""Tai cac video giao thong Viet Nam dung de thu nghiem (Pexels, mien phi, ban 720p) vao Data/videos/."""
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Data" / "videos"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"

VIDEOS = {
    "vn_giao_thong_ban_ngay.mp4": 33383214,
    "vn_nga_tu_xe_may_o_to.mp4": 30247347,
    "vn_hcm_nga_tu_ban_dem.mp4": 33486917,
    "vn_ket_xe_xe_may_o_to.mp4": 39221451,
    "vn_hanoi_pho_xe_may_xe_buyt.mp4": 33383212,
    "vn_cao_toc_flycam.mp4": 32971521,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name, vid in VIDEOS.items():
        dest = OUT / name
        if dest.exists():
            print(f"[co san] {name}")
            continue
        url = f"https://www.pexels.com/download/video/{vid}/?h=720&w=1280"
        try:
            with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            print(f"[da tai] {name} ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:  # mang/nguon loi: bao lai, tai tay theo link trong README
            dest.unlink(missing_ok=True)
            failed += 1
            print(f"[loi] {name}: {e} -> tai tay tai https://www.pexels.com/video/{vid}/")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
