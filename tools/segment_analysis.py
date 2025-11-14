"""分段分析与示例输出。

文件说明:
    复现论文的分段差时分析思路：将赛道划分为若干片段，
    统计策略在每片段的用时与进度，输出简单文本报告。

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


def run_segments(env: DroneRaceEnv, agent: DummyAgent) -> None:
    """按片段运行并统计用时。"""
    obs = env.reset()
    seg_times = []
    current_gate = int(obs["gate_id"])  # 起始门
    steps_in_seg = 0

    for _ in range(4000):
        action = agent.act(obs)
        obs, r, done, info = env.step(action)
        steps_in_seg += 1
        if info.get("gate_crossed"):
            seg_times.append(steps_in_seg * env.cfg.dt)
            steps_in_seg = 0
        if done:
            break

    if steps_in_seg > 0:
        seg_times.append(steps_in_seg * env.cfg.dt)

    print("Segment count:", len(seg_times))
    print("Mean time per segment:", np.mean(seg_times) if seg_times else 0.0)
    print("Segment times (first 10):", seg_times[:10])


def main() -> None:
    """入口。"""
    env = DroneRaceEnv()
    agent = DummyAgent(accel_gain=3.0, vel_damp=0.8)
    run_segments(env, agent)


if __name__ == "__main__":
    main()
