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
        fixed_object_position = env_cfg.fixed_object_position,
        floor_penalty = env_cfg.floor_penalty,
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
        # Stateful policies (e.g. BC with previous-action augmentation) opt in by
        # exposing a `reset()` method so episode boundaries clear their state.
        reset_fn = getattr(policy_fn, "reset", None)
        if callable(reset_fn):
            reset_fn()
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


def load_expert_transitions(
    npz_path: str | Path,
    include_prev_action: bool = True,
    frame_stack: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load expert transitions, optionally augmenting each observation with the
    previous action and a stack of the most recent env observations.

    Each augmented observation has the layout:

        [ obs_{t-K+1}, obs_{t-K+2}, ..., obs_{t-1}, obs_t, prev_action_t ]

    where `K = frame_stack` and indices before the start of the trajectory
    are zero-padded. `prev_action_t = actions[t-1]` (zeros at t=0).

    Frame stacking lets the BC network recover the expert FSM's elapsed-time
    counters from a window of recent obs (e.g. "I've been stationary for K
    steps -> commit to descend / close"), which a single obs cannot encode.
    The prev-action augmentation keeps the policy from flipping its gripper
    sign on adjacent steps.
    """
    frame_stack = int(max(1, frame_stack))
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

        T, env_obs_dim = obs.shape
        action_dim = acts.shape[1]

        if frame_stack > 1:
            stacked = np.zeros((T, frame_stack * env_obs_dim), dtype = np.float32)
            for t in range(T):
                start = max(0, t - frame_stack + 1)
                window = obs[start : t + 1]
                pad_count = frame_stack - window.shape[0]
                if pad_count > 0:
                    pad = np.zeros((pad_count, env_obs_dim), dtype = np.float32)
                    window = np.concatenate([pad, window], axis = 0)
                stacked[t] = window.reshape(-1)
            obs_out = stacked
        else:
            obs_out = obs

        if include_prev_action:
            zero_first = np.zeros((1, action_dim), dtype = np.float32)
            prev_acts = np.concatenate([zero_first, acts[:-1]], axis = 0)
            obs_out = np.concatenate([obs_out, prev_acts], axis = 1)

        obs_buffer.append(obs_out)
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
