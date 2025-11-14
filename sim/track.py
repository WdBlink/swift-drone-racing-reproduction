"""竞速赛道与闸门模型。

文件说明:
    定义 FPV 竞速赛道的闸门 `Gate` 与赛道 `RaceTrack`。
    提供默认 7 方形闸门布局（近似论文赛道，单圈约 75m）。

作者:
    wdblink
"""

from dataclasses import dataclass
from typing import List, Tuple
import math


@dataclass
class Gate:
    """赛道闸门。

    说明:
        使用一个空间位置与法向量来近似方形闸门的穿越判定。

    属性:
        gate_id: 闸门序号。
        position: 闸门中心位置 `(x, y, z)`，单位米。
        normal: 闸门法向量 `(nx, ny, nz)`，指向穿越方向。
        size: 闸门边长，单位米。
        tolerance: 穿越容差，闸门厚度近似判定，单位米。
    """

    gate_id: int
    position: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    size: float = 2.0
    tolerance: float = 0.5

    def contains(self, p: Tuple[float, float, float]) -> bool:
        """判断点是否在闸门面片范围内（基于闸门局部坐标）。

        Args:
            p: 空间点 `(x, y, z)`。

        Returns:
            是否位于闸门面片的局部方形范围内。
        """
        cx, cy, cz = self.position
        nx, ny, nz = self.normal

        def norm(a):
            return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

        def dot(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

        def cross(a, b):
            return (
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0],
            )

        n = (nx, ny, nz)
        n_norm = norm(n) + 1e-6
        n = (n[0]/n_norm, n[1]/n_norm, n[2]/n_norm)

        world_up = (0.0, 0.0, 1.0)
        u = cross(n, world_up)
        if norm(u) < 1e-6:
            world_alt = (1.0, 0.0, 0.0)
            u = cross(n, world_alt)
        u_norm = norm(u) + 1e-6
        u = (u[0]/u_norm, u[1]/u_norm, u[2]/u_norm)
        v = cross(n, u)
        v_norm = norm(v) + 1e-6
        v = (v[0]/v_norm, v[1]/v_norm, v[2]/v_norm)

        dx = (p[0]-cx, p[1]-cy, p[2]-cz)
        du = dot(dx, u)
        dv = dot(dx, v)
        dn = abs(dot(dx, n))

        return (abs(du) <= self.size/2.0) and (abs(dv) <= self.size/2.0) and (dn <= self.tolerance)

    def crossed(self, p_prev: Tuple[float, float, float], p: Tuple[float, float, float]) -> bool:
        """判断从上一位置到当前位置是否“穿越”闸门平面。

        Args:
            p_prev: 上一位置。
            p: 当前位置。

        Returns:
            是否发生穿越（线段与闸门平面相交且落在闸门开口范围内）。
        """
        nx, ny, nz = self.normal
        cx, cy, cz = self.position
        def dot(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

        d1 = dot((p_prev[0]-cx, p_prev[1]-cy, p_prev[2]-cz), (nx, ny, nz))
        d2 = dot((p[0]-cx, p[1]-cy, p[2]-cz), (nx, ny, nz))
        if d1 == 0.0 and d2 == 0.0:
            return self.contains(p)
        if (d1 <= 0.0 and d2 >= 0.0) or (d1 >= 0.0 and d2 <= 0.0):
            t = d1 / (d1 - d2 + 1e-6)
            ix = p_prev[0] + t * (p[0] - p_prev[0])
            iy = p_prev[1] + t * (p[1] - p_prev[1])
            iz = p_prev[2] + t * (p[2] - p_prev[2])
            return self.contains((ix, iy, iz))
        return False


class RaceTrack:
    """竞速赛道。"""

    def __init__(self, gates: List[Gate]):
        self.gates = gates

    @staticmethod
    def default_swift_track() -> "RaceTrack":
        gates: List[Gate] = []
        gates.append(Gate(1, (0.0, 0.0, 1.5), (1.0, 0.0, 0.0), size=2.0))
        gates.append(Gate(2, (10.0, 5.0, 2.0), (0.0, 1.0, 0.0), size=2.0))
        gates.append(Gate(3, (18.0, -3.0, 2.5), (-1.0, 0.0, 0.0), size=2.0))
        gates.append(Gate(4, (25.0, 0.0, 6.0), (0.0, -1.0, 0.0), size=2.0))
        gates.append(Gate(5, (25.0, 0.0, 1.5), (0.0, -1.0, 0.0), size=2.0))
        gates.append(Gate(6, (15.0, -10.0, 2.0), (-1.0, 0.0, 0.0), size=2.0))
        gates.append(Gate(7, (5.0, 0.0, 1.5), (1.0, 0.0, 0.0), size=2.0))
        return RaceTrack(gates)

    def next_gate_id(self, current_gate_id: int) -> int:
        idx = None
        for i, g in enumerate(self.gates):
            if g.gate_id == current_gate_id:
                idx = i
                break
        if idx is None:
            return self.gates[0].gate_id
        return self.gates[(idx + 1) % len(self.gates)].gate_id

    def get_gate(self, gate_id: int) -> Gate:
        for g in self.gates:
            if g.gate_id == gate_id:
                return g
        return self.gates[0]
