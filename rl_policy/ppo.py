"""PPO 算法实现（最小可用版）。

文件说明:
    实现基于 GAE 的 on-policy PPO（clip 目标），使用 MLP 策略-价值网络，
    以 `DroneRaceEnv` 为后端环境进行训练。该实现旨在验证与占位，
    后续可替换为更高性能的分布式版本（如 RLlib / SB3）。

作者:
    wdblink
"""

import os
import sys
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim.env import DroneRaceEnv
from rl_policy.env_wrapper import obs_to_vec


def mlp(in_dim: int, hidden: int, out_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """初始化两层 MLP 参数（ReLU）。"""
    W1 = np.random.randn(hidden, in_dim).astype(np.float32) * 0.05
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = np.random.randn(out_dim, hidden).astype(np.float32) * 0.05
    b2 = np.zeros(out_dim, dtype=np.float32)
    return W1, b1, W2, b2


def forward_mlp(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """前向计算。"""
    h = np.maximum(0.0, W1 @ x + b1)
    y = W2 @ h + b2
    return y, h


@dataclass
class PPOConfig:
    """PPO 配置。"""
    horizon: int = 2048
    batch_size: int = 256
    epochs: int = 10
    gamma: float = 0.995
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 3e-4
    ent_coef: float = 0.0
    hidden: int = 128
    act_sigma: float = 2.0


class PPOAgent:
    """最小 PPO Agent（连续动作，高斯策略，固定对角协方差）。"""

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig = PPOConfig()):
        self.cfg = cfg
        self.W1_pi, self.b1_pi, self.W2_pi, self.b2_pi = mlp(obs_dim, cfg.hidden, act_dim)
        self.W1_v, self.b1_v, self.W2_v, self.b2_v = mlp(obs_dim, cfg.hidden, 1)
        self.sigma = float(cfg.act_sigma)

    def policy_forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu, h = forward_mlp(x, self.W1_pi, self.b1_pi, self.W2_pi, self.b2_pi)
        return mu.astype(np.float32), h

    def value_forward(self, x: np.ndarray) -> float:
        v, _ = forward_mlp(x, self.W1_v, self.b1_v, self.W2_v, self.b2_v)
        return float(v.squeeze())

    def sample_action(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        mu, _ = self.policy_forward(x)
        a = mu + np.random.randn(*mu.shape).astype(np.float32) * self.sigma
        logp = -0.5 * np.sum(((a - mu) ** 2) / (self.sigma ** 2) + np.log(2 * math.pi * (self.sigma ** 2)))
        return a.astype(np.float32), float(logp)

    def logprob(self, x: np.ndarray, a: np.ndarray) -> float:
        mu, _ = self.policy_forward(x)
        return -0.5 * np.sum(((a - mu) ** 2) / (self.sigma ** 2) + np.log(2 * math.pi * (self.sigma ** 2)))

    def update(self, X: np.ndarray, A: np.ndarray, ADV: np.ndarray, LOGP_OLD: np.ndarray, RET: np.ndarray) -> None:
        """一次 PPO 更新（多轮 epoch）。"""
        cfg = self.cfg
        N = X.shape[0]
        idx = np.arange(N)
        for ep in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, N, cfg.batch_size):
                batch = idx[start:start+cfg.batch_size]
                x = X[batch]
                a = A[batch]
                adv = ADV[batch]
                logp_old = LOGP_OLD[batch]
                ret = RET[batch]

                grad_W1_pi = np.zeros_like(self.W1_pi)
                grad_b1_pi = np.zeros_like(self.b1_pi)
                grad_W2_pi = np.zeros_like(self.W2_pi)
                grad_b2_pi = np.zeros_like(self.b2_pi)

                grad_W1_v = np.zeros_like(self.W1_v)
                grad_b1_v = np.zeros_like(self.b1_v)
                grad_W2_v = np.zeros_like(self.W2_v)
                grad_b2_v = np.zeros_like(self.b2_v)

                for i in range(x.shape[0]):
                    xi = x[i]
                    ai = a[i]
                    hi_mu, hpi = forward_mlp(xi, self.W1_pi, self.b1_pi, self.W2_pi, self.b2_pi)
                    mu = hi_mu.astype(np.float32)
                    logp = -0.5 * np.sum(((ai - mu) ** 2) / (self.sigma ** 2) + np.log(2 * math.pi * (self.sigma ** 2)))
                    ratio = math.exp(logp - logp_old[i])
                    surr1 = ratio * adv[i]
                    surr2 = np.clip(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv[i]
                    pi_loss = -min(surr1, surr2)

                    dlogp_dmu = (ai - mu) / (self.sigma ** 2)
                    coeff = -1.0 if surr1 <= surr2 else 0.0
                    dL_dmu = coeff * dlogp_dmu * adv[i]

                    dL_dy = dL_dmu
                    grad_W2_pi += np.outer(dL_dy, hpi)
                    grad_b2_pi += dL_dy
                    dh = (self.W2_pi.T @ dL_dy) * (hpi > 0).astype(np.float32)
                    grad_W1_pi += np.outer(dh, xi)
                    grad_b1_pi += dh

                    vi, hvi = forward_mlp(xi, self.W1_v, self.b1_v, self.W2_v, self.b2_v)
                    vi = float(vi.squeeze())
                    diff = vi - float(ret[i])
                    dL_v = 2.0 * diff
                    grad_W2_v += np.outer(np.array([dL_v], dtype=np.float32), hvi)
                    grad_b2_v += np.array([dL_v], dtype=np.float32)
                    dhv = (self.W2_v.T @ np.array([dL_v], dtype=np.float32)).reshape(-1) * (hvi > 0).astype(np.float32)
                    grad_W1_v += np.outer(dhv, xi)
                    grad_b1_v += dhv

                self.W1_pi -= cfg.pi_lr * grad_W1_pi / x.shape[0]
                self.b1_pi -= cfg.pi_lr * grad_b1_pi / x.shape[0]
                self.W2_pi -= cfg.pi_lr * grad_W2_pi / x.shape[0]
                self.b2_pi -= cfg.pi_lr * grad_b2_pi / x.shape[0]

                self.W1_v -= cfg.vf_lr * grad_W1_v / x.shape[0]
                self.b1_v -= cfg.vf_lr * grad_b1_v / x.shape[0]
                self.W2_v -= cfg.vf_lr * grad_W2_v / x.shape[0]
                self.b2_v -= cfg.vf_lr * grad_b2_v / x.shape[0]


def rollout(env: DroneRaceEnv, agent: PPOAgent, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """采样一个 on-policy 批次。"""
    X = []
    A = []
    R = []
    LOGP = []
    V = []
    obs = env.reset()
    for t in range(horizon):
        x = obs_to_vec(obs)
        a, logp = agent.sample_action(x)
        obs, r, done, info = env.step(a)
        X.append(x)
        A.append(a)
        R.append(r)
        LOGP.append(logp)
        V.append(agent.value_forward(x))
        if done:
            obs = env.reset()
    return np.array(X, dtype=np.float32), np.array(A, dtype=np.float32), np.array(R, dtype=np.float32), np.array(LOGP, dtype=np.float32), np.array(V, dtype=np.float32)


def compute_gae(R: np.ndarray, V: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """计算 GAE 与目标回报。"""
    T = len(R)
    ADV = np.zeros(T, dtype=np.float32)
    RET = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    next_v = 0.0
    for t in reversed(range(T)):
        delta = R[t] + gamma * (next_v) - V[t]
        last_gae = delta + gamma * lam * last_gae
        ADV[t] = last_gae
        RET[t] = ADV[t] + V[t]
        next_v = V[t]
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-6)
    return ADV, RET


def train_ppo(episodes: int = 10) -> PPOAgent:
    """PPO 训练入口。"""
    env = DroneRaceEnv()
    obs_dim = len(obs_to_vec(env.reset()))
    act_dim = 3
    cfg = PPOConfig(horizon=2048, batch_size=256, epochs=5, hidden=128)
    agent = PPOAgent(obs_dim, act_dim, cfg)
    for ep in range(episodes):
        X, A, R, LOGP, V = rollout(env, agent, cfg.horizon)
        ADV, RET = compute_gae(R, V, cfg.gamma, cfg.lam)
        agent.update(X, A, ADV, LOGP, RET)
        print(f"[PPO] ep={ep+1} meanR={float(np.mean(R)):.3f} gates? (see tools) batch={len(R)}")
    return agent


def main() -> None:
    """入口。"""
    agent = train_ppo(episodes=5)


if __name__ == "__main__":
    main()
