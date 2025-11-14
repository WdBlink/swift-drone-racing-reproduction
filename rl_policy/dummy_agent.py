"""占位策略: 线性导航加速度控制。

文件说明:
    为验证仿真环境骨架可运行性，提供一个简单的策略：
    沿当前闸门方向加速，并加入速度阻尼以避免发散。

作者:
    wdblink
"""

import numpy as np


class DummyAgent:
    """占位策略类。

    说明:
        根据观测 `to_gate` 与当前速度 `vel` 计算加速度指令。
    """

    def __init__(self, accel_gain: float = 2.0, vel_damp: float = 1.0):
        """构造函数。

        Args:
            accel_gain: 朝向目标的加速度增益。
            vel_damp: 速度阻尼系数。
        """
        self.accel_gain = accel_gain
        self.vel_damp = vel_damp

    def act(self, obs: dict) -> np.ndarray:
        """计算动作。

        Args:
            obs: 观测字典，至少包含 `to_gate` 与 `vel`。

        Returns:
            加速度向量 `(ax, ay, az)`。
        """
        to_gate = obs["to_gate"].astype(np.float32)
        vel = obs["vel"].astype(np.float32)
        a = self.accel_gain * to_gate / (np.linalg.norm(to_gate) + 1e-6) - self.vel_damp * vel
        return a
