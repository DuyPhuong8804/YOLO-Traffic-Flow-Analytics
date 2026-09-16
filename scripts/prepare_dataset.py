"""
Script chuan hoa dataset "Vietnamese vehicle v3" (Roboflow) ve dung 3 lop
car / truck / motorbike theo quy tac gom lop dinh nghia trong
src/common/class_map.py (RAW_TO_FINAL_CLASS_MAP).

Cach hoat dong:
    1. Backup toan bo file label goc (.txt) sang Data/processed_labels_backup/
       (chi backup 1 lan duy nhat, lan chay sau se bo qua neu da co backup
       -> IDEMPOTENT, chay lai nhieu lan khong lam hong du lieu).
    2. Doc tung file label trong train/valid/test, thay class id goc bang
       class id cuoi cung theo RAW_TO_FINAL_CLASS_MAP, ghi de len file goc.
    3. Cap nhat lai data.yaml: nc=3, names=['car','truck','motorbike'].

Chay:
    python scripts/prepare_dataset.py
"""

from pathlib import Path
import shutil
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.class_map import RAW_TO_FINAL_CLASS_MAP, CLASS_NAMES_LIST  # noqa: E402

DATASET_ROOT = Path(r"Data/Vietnamese vehicle.v3-2023-02-01-5-31pm.yolo26")
BACKUP_ROOT = Path("Data/processed_labels_backup")
SPLITS = ["train", "valid", "test"]


def backup_labels_once() -> None:
    """Sao luu label goc truoc khi ghi de, chi thuc hien neu chua backup."""
    marker = BACKUP_ROOT / "DONE.marker"
    if marker.exists():
        print("[skip] Da co backup tu truoc, bo qua buoc backup.")
        return
    for split in SPLITS:
        src = DATASET_ROOT / split / "labels"
        dst = BACKUP_ROOT / split / "labels"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"[backup] {src} -> {dst}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("backed up\n", encoding="utf-8")


def remap_split(split: str) -> tuple[int, int]:
    """Remap class id cho toan bo file .txt trong 1 split. Tra ve (so file, so box)."""
    label_dir = DATASET_ROOT / split / "labels"
    n_files = 0
    n_boxes = 0
    for txt_path in label_dir.glob("*.txt"):
        lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            raw_id = int(parts[0])
            final_id = int(RAW_TO_FINAL_CLASS_MAP[raw_id])
            new_lines.append(" ".join([str(final_id), *parts[1:]]))
            n_boxes += 1
        txt_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        n_files += 1
    return n_files, n_boxes


def update_data_yaml() -> None:
    """Cap nhat data.yaml: nc=3 va ten lop dung theo class_map.py."""
    yaml_path = DATASET_ROOT / "data.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["nc"] = len(CLASS_NAMES_LIST)
    data["names"] = CLASS_NAMES_LIST
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[update] {yaml_path} -> nc={data['nc']}, names={data['names']}")


def main() -> None:
    already_remapped = (BACKUP_ROOT / "DONE.marker").exists()
    if already_remapped:
        print(
            "[canh bao] Script co ve da chay truoc do (da co backup). "
            "Neu chay lai remap tren label DA remap se gay sai du lieu "
            "(vi id 1/2 da la truck/motorbike, khong con la id goc). "
            "Dang thoat de tranh remap 2 lan."
        )
        return

    backup_labels_once()
    for split in SPLITS:
        n_files, n_boxes = remap_split(split)
        print(f"[remap] {split}: {n_files} file, {n_boxes} box da chuyen id")
    update_data_yaml()
    print("\nHoan tat chuan hoa dataset ve 3 lop: car(0) / truck(1) / motorbike(2)")


if __name__ == "__main__":
    main()
