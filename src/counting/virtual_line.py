"""
Dinh nghia duong ke ao (virtual counting line) va cac phep toan hinh hoc lien quan:
    - kiem tra 2 doan thang co giao nhau khong (segment intersection, thuat toan CCW)
    - xac dinh huong di chuyen dua tren vector phap tuyen cua duong ke

Day la module THUAN HINH HOC, khong phu thuoc model/video -> de unit test doc lap
(xem tests/test_counting_logic.py).
"""

from dataclasses import dataclass

Point = tuple[float, float]


def _ccw(a: Point, b: Point, c: Point) -> float:
    """Dinh huong cua 3 diem a,b,c: >0 nguoc chieu kim dong ho, <0 cung chieu, =0 thang hang."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Kiem tra doan thang (p1,p2) co giao thuc su voi doan thang (p3,p4) khong.
    Thuat toan orientation/CCW chuan - dung cho ca truong hop giao "cham" (touching)."""
    d1 = _ccw(p3, p4, p1)
    d2 = _ccw(p3, p4, p2)
    d3 = _ccw(p1, p2, p3)
    d4 = _ccw(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True

    # Truong hop dac biet: diem nam dung tren doan thang (thang hang, d=0)
    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    if d1 == 0 and on_segment(p3, p1, p4):
        return True
    if d2 == 0 and on_segment(p3, p2, p4):
        return True
    if d3 == 0 and on_segment(p1, p3, p2):
        return True
    if d4 == 0 and on_segment(p1, p4, p2):
        return True
    return False


@dataclass
class VirtualLine:
    """1 duong ke ao dung de dem xe cat qua, gan voi 1 camera."""

    name: str
    point1: Point
    point2: Point
    direction_a_label: str
    direction_b_label: str

    def normal_vector(self) -> Point:
        """Vector phap tuyen (vuong goc voi duong ke), dung de xac dinh huong di chuyen."""
        lx = self.point2[0] - self.point1[0]
        ly = self.point2[1] - self.point1[1]
        return (-ly, lx)

    def crossed_by(self, prev_center: Point, curr_center: Point) -> bool:
        """True neu doan di chuyen (prev_center -> curr_center) cat qua duong ke nay."""
        return segments_intersect(prev_center, curr_center, self.point1, self.point2)

    def direction_label(self, prev_center: Point, curr_center: Point) -> str:
        """Xac dinh nhan huong di chuyen (direction_a/b) dua vao dau cua tich vo huong
        giua vector di chuyen va vector phap tuyen cua duong ke."""
        nx, ny = self.normal_vector()
        vx = curr_center[0] - prev_center[0]
        vy = curr_center[1] - prev_center[1]
        dot = vx * nx + vy * ny
        return self.direction_a_label if dot >= 0 else self.direction_b_label
