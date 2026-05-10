import json
import random
from pathlib import Path
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from envs.fetch_throw_env import FetchThrowWrapper
from scripts.experiment_config import EnvironmentConfig


def reseed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_throw_env(env_cfg: EnvironmentConfig, seed: int | None = None):
    base_env = gym.make(
        env_cfg.env_id,
        max_episode_steps = env_cfg.max_episode_steps,
    )
    env = FetchThrowWrapper(
        base_env,
        throw_overclock_factor = env_cfg.throw_overclock_factor,
        override_reward_on_score = env_cfg.override_reward_on_score,
        score_reward = env_cfg.score_reward,
        goal_bonus = env_cfg.goal_bonus,
        terminate_on_score = env_cfg.terminate_on_score,
        terminate_ball_below_z = env_cfg.terminate_ball_below_z,
    )
    if seed is not None:
        env.reset(seed = seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def flatten_obs(obs_dict: dict) -> np.ndarray:
    return np.asarray(obs_dict["observation"], dtype = np.float32)


def evaluate_policy_callable(
    policy_fn: Callable[[dict], np.ndarray],
    env_cfg: EnvironmentConfig,
    seed: int,
    n_episodes: int,
) -> dict:
    env = make_throw_env(env_cfg, seed = seed)
    returns = []
    successes = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed = seed + ep)
        done = False
        truncated = False
        ep_return = 0.0
        while not (done or truncated):
            action = policy_fn(obs)
            action = np.asarray(action, dtype = np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, truncated, info = env.step(action)
            ep_return += float(reward)
        returns.append(ep_return)
        successes.append(float(info.get("is_success", 0.0)))

    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
    }


def load_expert_transitions(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    dataset = np.load(npz_path, allow_pickle = True)
    trajectories = dataset["trajectories"]
    obs_buffer = []
    act_buffer = []
    for traj in trajectories:
        obs = np.asarray(traj["obs"], dtype = np.float32)
        acts = np.asarray(traj["actions"], dtype = np.float32)
        assert obs.ndim == 2, "Expected 2D observations per trajectory."
        assert acts.ndim == 2, "Expected 2D actions per trajectory."
        assert obs.shape[0] == acts.shape[0], "Obs/action step mismatch in trajectory."
        obs_buffer.append(obs)
        act_buffer.append(acts)

    all_obs = np.concatenate(obs_buffer, axis = 0)
    all_actions = np.concatenate(act_buffer, axis = 0)
    return all_obs, all_actions


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents = True, exist_ok = True)
    return p


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding = "utf-8") as f:
        json.dump(payload, f, indent = 2)


def save_curve_plot(x, y, title: str, xlabel: str, ylabel: str, out_path: str | Path):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    plt.figure(figsize = (8, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi = 150)
    plt.close()
