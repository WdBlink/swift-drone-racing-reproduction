"""VIO 存根实现（仿真）。

文件说明:
    提供简化的视觉惯性里程计算法的仿真版本，
    在已知真实状态的情况下加入经验噪声与延迟，
    用于构建 Sim2Real 的感知噪声模型骨架。

作者:
    wdblink
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class VIONoiseConfig:
    """VIO 噪声与延迟配置。"""
    pos_sigma: float = 0.05
    vel_sigma: float = 0.1
    yaw_sigma_deg: float = 1.0
    latency_s: float = 0.02


class VIOStub:
    """仿真 VIO 管线存根。"""

    def __init__(self, cfg: Optional[VIONoiseConfig] = None):
        """构造函数。"""
        self.cfg = cfg or VIONoiseConfig()
        self._queue: list[Tuple[np.ndarray, np.ndarray, float]] = []

    def push_truth(self, p: np.ndarray, v: np.ndarray, yaw_rad: float, t: float) -> None:
        """推入真实状态（用于模拟延迟队列）。"""
        self._queue.append((p.copy(), v.copy(), yaw_rad))
        while len(self._queue) > max(1, int(self.cfg.latency_s / 0.02)):
            self._queue.pop(0)

    def estimate(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """输出带噪估计的位姿与速度。"""
        if not self._queue:
            return np.zeros(3), np.zeros(3), 0.0
        p, v, yaw = self._queue[0]
        p_est = p + np.random.randn(3) * self.cfg.pos_sigma
        v_est = v + np.random.randn(3) * self.cfg.vel_sigma
        yaw_est = yaw + np.deg2rad(np.random.randn() * self.cfg.yaw_sigma_deg)
        return p_est.astype(np.float32), v_est.astype(np.float32), float(yaw_est)
