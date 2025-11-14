"""仿真运行脚本。

文件说明:
    使用占位策略 `DummyAgent` 在 `DroneRaceEnv` 中执行若干步，
    输出基本统计信息，验证环境可用性。

作者:
    wdblink
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim.env import DroneRaceEnv
from rl_policy.dummy_agent import DummyAgent


def main() -> None:
    """运行占位策略仿真并打印统计。"""
    env = DroneRaceEnv()
    agent = DummyAgent(accel_gain=3.0, vel_damp=0.8)
    obs = env.reset()
    total_reward = 0.0
    gates = 0
    for _ in range(3000):
        action = agent.act(obs)
        obs, r, done, info = env.step(action)
        total_reward += r
        if info.get("gate_crossed"):
            gates += 1
        if done:
            break
    print(f"steps={env.step_count} total_reward={total_reward:.2f} gates_crossed={gates}")


if __name__ == "__main__":
    main()
