"""环境观测封装与归一化。

文件说明:
    提供将 `DroneRaceEnv` 的字典观测转换为连续向量的工具，
    并进行简单的尺度归一化，方便策略网络输入。

作者:
    wdblink
"""

import numpy as np
from typing import Dict


def obs_to_vec(obs: Dict) -> np.ndarray:
    """将字典观测转换为向量。"""
    to_gate = obs["to_gate"].reshape(3)
    to_next = obs["to_next"].reshape(3)
    vel = obs["vel"].reshape(3)
    yaw = np.array([obs.get("yaw", 0.0)], dtype=np.float32)
    conf_gate = np.array([obs.get("conf_gate", 1.0)], dtype=np.float32)
    conf_next = np.array([obs.get("conf_next", 1.0)], dtype=np.float32)
    gate_id = np.array([float(obs.get("gate_id", 1))], dtype=np.float32)
    vec = np.concatenate([
        to_gate / (np.linalg.norm(to_gate) + 1e-6),
        to_next / (np.linalg.norm(to_next) + 1e-6),
        vel / (np.linalg.norm(vel) + 1e-6 + 1.0),
        yaw,
        conf_gate,
        conf_next,
        gate_id / 7.0,
    ]).astype(np.float32)
    return vec
