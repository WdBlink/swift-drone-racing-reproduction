"""可视化评测与图表输出。

文件说明:
    生成基础图表：
    - 每步奖励曲线
    - 分段用时柱状图
    - 速度与对齐度随时间曲线

作者:
    wdblink
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim.env import DroneRaceEnv
from rl_policy.dummy_agent import DummyAgent


def collect(env: DroneRaceEnv, agent: DummyAgent, steps: int = 2000):
    """采集轨迹与奖励序列。"""
    obs = env.reset()
    rewards = []
    aligns = []
    speeds = []
    seg_times = []
    steps_in_seg = 0
    for _ in range(steps):
        a = agent.act(obs)
        obs, r, done, info = env.step(a)
        rewards.append(r)
        v = obs["vel"]
        gate_vec = obs["to_gate"]
        align = float(np.dot(v/(np.linalg.norm(v)+1e-6), gate_vec/(np.linalg.norm(gate_vec)+1e-6)))
        aligns.append(align)
        speeds.append(float(np.linalg.norm(v)))
        steps_in_seg += 1
        if info.get("gate_crossed"):
            seg_times.append(steps_in_seg * env.cfg.dt)
            steps_in_seg = 0
        if done:
            break
    if steps_in_seg > 0:
        seg_times.append(steps_in_seg * env.cfg.dt)
    return np.array(rewards), np.array(aligns), np.array(speeds), np.array(seg_times)


def save_plots(rewards, aligns, speeds, seg_times, out_dir: str = "out"):
    """保存图表。"""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 3))
    plt.plot(rewards)
    plt.title("Reward per step")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(aligns, label="align")
    plt.plot(speeds, label="speed")
    plt.legend()
    plt.title("Align and Speed")
    plt.xlabel("step")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "align_speed.png"))
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.bar(np.arange(len(seg_times)), seg_times)
    plt.title("Segment times")
    plt.xlabel("segment idx")
    plt.ylabel("time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "segment_times.png"))
    plt.close()


def main() -> None:
    """入口。"""
    env = DroneRaceEnv()
    agent = DummyAgent(accel_gain=3.0, vel_damp=0.8)
    rewards, aligns, speeds, seg_times = collect(env, agent, steps=3000)
    save_plots(rewards, aligns, speeds, seg_times, out_dir="out")
    print("saved plots to ./out")


if __name__ == "__main__":
    main()
