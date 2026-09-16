"""
Script ho tro xac minh truc quan class id trong dataset raw truoc khi remap.
Lay vai anh mau CHI chua 1 class duy nhat cho tung id, ve bounding box, luu ra
Data/class_inspection/ de xem bang mat va doi chieu voi src/common/class_map.py.

Chi can chay lai khi doi sang dataset khac / nghi ngo mapping sai.

Chay:
    python scripts/inspect_classes.py
"""

from pathlib import Path
import random

from PIL import Image, ImageDraw

DATASET_ROOT = Path(r"Data/Vietnamese vehicle.v3-2023-02-01-5-31pm.yolo26/train")
OUT_DIR = Path("Data/class_inspection")
N_CLASSES = 4  # so lop GOC truoc khi remap
SAMPLES_PER_CLASS = 2

random.seed(42)


def main() -> None:
    img_dir = DATASET_ROOT / "images"
    lbl_dir = DATASET_ROOT / "labels"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_class: dict[int, list[Path]] = {c: [] for c in range(N_CLASSES)}
    for lbl_file in lbl_dir.glob("*.txt"):
        lines = [l.strip() for l in lbl_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        classes_in_file = {int(l.split()[0]) for l in lines}
        if len(classes_in_file) == 1:
            by_class[next(iter(classes_in_file))].append(lbl_file)

    for cid, files in by_class.items():
        print(f"Class {cid}: {len(files)} anh chi chua 1 class duy nhat")
        sample = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))
        for i, lbl_file in enumerate(sample):
            img_path = img_dir / (lbl_file.stem + ".jpg")
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            draw = ImageDraw.Draw(img)
            for line in lbl_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                _, xc, yc, bw, bh = line.split()
                xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
                x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
                x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            out_path = OUT_DIR / f"class{cid}_sample{i}.jpg"
            img.save(out_path)
            print("  saved", out_path)


if __name__ == "__main__":
    main()
