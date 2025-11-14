"""无人机竞速仿真环境骨架。

文件说明:
    提供 `DroneRaceEnv` 与其配置 `EnvConfig`，用于策略训练与基础验证。
    该骨架仅实现最小可跑通版本：简化动力学、门穿越判定、奖励与步进。

作者:
    wdblink
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np

from .track import RaceTrack
from perception import PerceptionFusion, FusionConfig


@dataclass
class EnvConfig:
    """环境配置。

    说明:
        控制频率、最大步数、奖励权重等基础参数。

    属性:
        dt: 控制步长（秒）。
        max_steps: 最大步数（单回合）。
        speed_limit: 速度上限（m/s）。
        accel_limit: 加速度上限（m/s^2）。
        gate_reward: 穿越闸门奖励。
        crash_penalty: 撞击惩罚。
        progress_scale: 进度奖励缩放。
    """

    dt: float = 0.02
    max_steps: int = 5000
    speed_limit: float = 25.0
    accel_limit: float = 30.0
    gate_reward: float = 10.0
    crash_penalty: float = -20.0
    progress_scale: float = 1.0
    control_cost_scale: float = 0.01
    align_reward_scale: float = 0.5
    lap_reward: float = 30.0


class DroneRaceEnv:
    """无人机竞速仿真环境（最小骨架）。

    说明:
        - 状态向量: 位置 `p`、速度 `v`、当前闸门 ID。
        - 动作向量: 期望加速度 `a`（世界系），裁剪至上限。
        - 奖励: 进度（朝向当前门的投影距离减少）、穿门奖励、撞击惩罚与越界终止。

    用途:
        作为策略训练与验证的起点。后续可替换为高保真动力学与视觉/IMU感知接入。
    """

    def __init__(self, cfg: Optional[EnvConfig] = None):
        """构造函数。

        Args:
            cfg: 环境配置，可为空则使用默认。
        """
        self.cfg = cfg or EnvConfig()
        self.track = RaceTrack.default_swift_track()
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """重置环境并返回初始观测。

        Returns:
            初始观测字典，包含位置、速度与目标闸门信息。
        """
        self.step_count = 0
        g1 = self.track.get_gate(1)
        self.p = np.array([g1.position[0] - 5.0, g1.position[1], g1.position[2]])
        self.v = np.zeros(3)
        self.current_gate_id = 1
        self.prev_p = self.p.copy()
        return self._observation()

    def _observation(self) -> Dict[str, Any]:
        """构建低维观测表示。

        Returns:
            观测字典：到当前/下一门相对向量、当前速度与门 ID。
        """
        gate = self.track.get_gate(self.current_gate_id)
        next_gate_id = self.track.next_gate_id(self.current_gate_id)
        next_gate = self.track.get_gate(next_gate_id)
        to_gate = np.array(gate.position) - self.p
        to_next = np.array(next_gate.position) - self.p
        if not hasattr(self, "perception"):
            self.perception = PerceptionFusion(FusionConfig())
        fused = self.perception.fuse(self.p, self.v, 0.0, to_gate, to_next)
        fused["gate_id"] = np.int32(self.current_gate_id)
        return fused

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """环境步进。

        Args:
            action: 期望加速度向量 `(ax, ay, az)`。

        Returns:
            (obs, reward, done, info)
        """
        self.step_count += 1
        a = np.clip(action.reshape(3), -self.cfg.accel_limit, self.cfg.accel_limit)
        self.prev_p = self.p.copy()
        self.v = np.clip(self.v + a * self.cfg.dt, -self.cfg.speed_limit, self.cfg.speed_limit)
        self.p = self.p + self.v * self.cfg.dt

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        gate = self.track.get_gate(self.current_gate_id)
        dist_prev = np.linalg.norm(np.array(gate.position) - self.prev_p)
        dist_now = np.linalg.norm(np.array(gate.position) - self.p)
        progress = (dist_prev - dist_now) * self.cfg.progress_scale
        reward += float(progress)

        if gate.crossed(tuple(self.prev_p), tuple(self.p)):
            reward += self.cfg.gate_reward
            self.current_gate_id = self.track.next_gate_id(self.current_gate_id)
            info["gate_crossed"] = True
            if self.current_gate_id == 1:
                reward += self.cfg.lap_reward
                info["lap_completed"] = True

        if not (-5.0 <= self.p[0] <= 35.0 and -20.0 <= self.p[1] <= 20.0 and 0.5 <= self.p[2] <= 8.5):
            reward += self.cfg.crash_penalty
            done = True
            info["crash"] = True

        v_norm = np.linalg.norm(self.v) + 1e-6
        to_gate_vec = np.array(gate.position) - self.p
        tg_norm = np.linalg.norm(to_gate_vec) + 1e-6
        align = float(np.dot(self.v / v_norm, to_gate_vec / tg_norm))
        reward += self.cfg.align_reward_scale * align

        control_cost = float(np.dot(a, a))
        reward -= self.cfg.control_cost_scale * control_cost

        if self.step_count >= self.cfg.max_steps:
            done = True

        return self._observation(), reward, done, info
