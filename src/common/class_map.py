"""
Dinh nghia tap trung 3 lop phuong tien cuoi cung dung xuyen suot du an
(training, tracking, counting, storage, API, dashboard).

QUY TAC GOM LOP (da xac minh bang truc quan tren tap Vietnamese vehicle v3):
    id goc 0 "car"       -> car        (giu nguyen)
    id goc 1 "motorbike"  -> motorbike  (giu nguyen)
    id goc 2 "truck"      -> truck      (giu nguyen)
    id goc 3 "bus"        -> truck      (GOP vao truck)

Ly do gop bus vao truck: trong thong ke luu luong giao thong, xe buyt va xe tai
duoc xep chung nhom "phuong tien co gioi co lon" (heavy vehicle) vi co dac diem
tuong dong ve kich thuoc, toc do di chuyen thap hon va muc do chiem dung lan
duong lon hon nhieu so voi xe con/xe may. Viec tach rieng bus thanh lop thu 4
khong nam trong yeu cau de bai (chi can car/truck/motorbike) va so luong mau
bus trong tap du lieu cung it (324/5795 box ~ 5.6%) nen khong anh huong nhieu
den chat luong mo hinh khi gop lop.
"""

from enum import IntEnum


class VehicleClass(IntEnum):
    """3 lop phuong tien cuoi cung, thu tu id dung trong data.yaml sau khi remap."""

    CAR = 0
    TRUCK = 1
    MOTORBIKE = 2


# Ten lop hien thi (tieng Viet) dung cho ve overlay / dashboard / bao cao
CLASS_NAMES_VI = {
    VehicleClass.CAR: "Xe con",
    VehicleClass.TRUCK: "Xe tai",
    VehicleClass.MOTORBIKE: "Xe may",
}

# Ten lop dung trong data.yaml / model (tieng Anh, khong dau, an toan cho moi tool)
CLASS_NAMES_EN = {
    VehicleClass.CAR: "car",
    VehicleClass.TRUCK: "truck",
    VehicleClass.MOTORBIKE: "motorbike",
}

# Danh sach ten theo dung thu tu index -> dung truc tiep cho data.yaml['names']
CLASS_NAMES_LIST = [CLASS_NAMES_EN[c] for c in sorted(VehicleClass)]

# Mau BGR (OpenCV) rieng cho tung lop khi ve bounding box overlay
CLASS_COLORS_BGR = {
    VehicleClass.CAR: (60, 180, 75),       # xanh la
    VehicleClass.TRUCK: (0, 130, 245),     # cam
    VehicleClass.MOTORBIKE: (230, 50, 50),  # xanh duong
}

# Mapping id lop GOC trong dataset "Vietnamese vehicle v3" (Roboflow) -> id lop CUOI CUNG
# 0=car, 1=motorbike, 2=truck, 3=bus (da xac minh truc quan, xem Data/class_inspection/)
RAW_TO_FINAL_CLASS_MAP = {
    0: VehicleClass.CAR,
    1: VehicleClass.MOTORBIKE,
    2: VehicleClass.TRUCK,
    3: VehicleClass.TRUCK,  # bus -> truck
}


def class_id_to_name(class_id: int) -> str:
    """Tra ve ten lop (EN) tu id, dung khi ghi DB / tra ve API."""
    return CLASS_NAMES_EN[VehicleClass(class_id)]
