CLASS_NAMES = ["car", "truck", "motorbike"]

CLASS_LABELS_VI = {
    "car": "Xe con",
    "truck": "Xe tải",
    "motorbike": "Xe máy",
}

# BGR cho OpenCV
CLASS_COLORS_BGR = {
    "car": (246, 130, 59),
    "truck": (11, 158, 245),
    "motorbike": (129, 185, 16),
}

# HEX cho bieu do (cung mau voi BGR ben tren)
CLASS_COLORS_HEX = {
    "car": "#3B82F6",
    "truck": "#F59E0B",
    "motorbike": "#10B981",
}

# Anh xa ten lop COCO -> 3 lop cua de tai, dung khi chay bang model pretrained
COCO_TO_PROJECT = {
    "car": "car",
    "truck": "truck",
    "bus": "truck",
    "motorcycle": "motorbike",
}


def to_project_class(name: str) -> str | None:
    if name in CLASS_NAMES:
        return name
    return COCO_TO_PROJECT.get(name)
