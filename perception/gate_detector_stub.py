"""闸门检测存根实现（仿真）。

文件说明:
    在仿真环境中，根据赛道已知的闸门位置，
    生成带噪的下一闸门相对向量与置信度。

作者:
    wdblink
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class DetectorNoiseConfig:
    """检测噪声与漏检配置。"""
    rel_sigma: float = 0.1
    drop_prob: float = 0.02


class GateDetectorStub:
    """仿真闸门检测存根。"""

    def __init__(self, cfg: Optional[DetectorNoiseConfig] = None):
        self.cfg = cfg or DetectorNoiseConfig()

    def detect(self, rel_vec: np.ndarray) -> Tuple[np.ndarray, float]:
        """对真实相对向量施加噪声与漏检。

        Args:
            rel_vec: 真实相对向量（门中心到机体）。

        Returns:
            (带噪相对向量, 置信度)
        """
        if np.random.rand() < self.cfg.drop_prob:
            return np.zeros(3, dtype=np.float32), 0.0
        noisy = rel_vec + np.random.randn(3) * self.cfg.rel_sigma
        conf = float(np.clip(1.0 - self.cfg.rel_sigma, 0.0, 1.0))
        return noisy.astype(np.float32), conf
