import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from torch.utils.data import DataLoader, TensorDataset

# Allow running as either `python -m scripts.train_ppo_bc_warmstart` or script path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.experiment_config import ExperimentConfig, make_dense_only_env_config
from scripts.training_utils import (
    ensure_dir,
    load_expert_transitions,
    make_throw_env,
    reseed,
    save_curve_plot,
    select_device,
    write_json,
)


class AugmentedThrowObservationWrapper(gym.Wrapper):
    """Expose the same flat input vector used by BC as the RL observation.

    Layout:
        [obs_{t-K+1}, ..., obs_t, prev_action_t]

    The BC model only worked once it had temporal context. PPO must therefore
    train in the same observation space if BC actor weights are used to
    initialize it.
    """

    def __init__(self, env: gym.Env, frame_stack: int, include_prev_action: bool):
        super().__init__(env)
        self.frame_stack = int(max(1, frame_stack))
        self.include_prev_action = bool(include_prev_action)

        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("Expected Fetch dict observations before augmentation.")
        raw_obs_space = env.observation_space["observation"]
        if not isinstance(raw_obs_space, spaces.Box):
            raise TypeError("Expected `observation` to be a Box space.")

        self.raw_obs_dim = int(raw_obs_space.shape[0])
        self.action_dim = int(env.action_space.shape[0])
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.obs_history: list[np.ndarray] = []

        low_parts = [np.tile(raw_obs_space.low.astype(np.float32), self.frame_stack)]
        high_parts = [np.tile(raw_obs_space.high.astype(np.float32), self.frame_stack)]
        if self.include_prev_action:
            low_parts.append(env.action_space.low.astype(np.float32))
            high_parts.append(env.action_space.high.astype(np.float32))

        self.observation_space = spaces.Box(
            low=np.concatenate(low_parts, axis=0),
            high=np.concatenate(high_parts, axis=0),
            dtype=np.float32,
        )

    def _augment(self, obs_dict: dict) -> np.ndarray:
        env_obs = np.asarray(obs_dict["observation"], dtype=np.float32)
        self.obs_history.append(env_obs)
        if len(self.obs_history) > self.frame_stack:
            self.obs_history = self.obs_history[-self.frame_stack :]

        pad_count = self.frame_stack - len(self.obs_history)
        if pad_count > 0:
            pad = [np.zeros(self.raw_obs_dim, dtype=np.float32)] * pad_count
            stacked = np.concatenate(pad + self.obs_history, axis=0)
        else:
            stacked = np.concatenate(self.obs_history, axis=0)

        if self.include_prev_action:
            return np.concatenate([stacked, self.prev_action], axis=0).astype(np.float32)
        return stacked.astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.obs_history = []
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.prev_action = np.asarray(action, dtype=np.float32).copy()
        return self._augment(obs), reward, terminated, truncated, info


class BCTanhActorCriticPolicy(ActorCriticPolicy):
    """PPO policy whose deterministic actor mean has BC's final tanh.

    SB3's default Gaussian policy uses an unconstrained linear mean. The BC
    policy learned a bounded action map (`Linear -> Tanh`), and this task is
    sensitive enough that removing the tanh can break rollout behavior even
    when supervised MSE looks small.
    """

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        linear_action_net = self.action_net
        self.action_net = nn.Sequential(linear_action_net, nn.Tanh()).to(self.device)
        # Recreate the optimizer so it sees the replaced action_net module.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


class PeriodicEvalSaveCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_fn,
        eval_every_steps: int,
        checkpoint_every_steps: int,
        out_dir: Path,
    ):
        super().__init__()
        self.eval_fn = eval_fn
        self.eval_every_steps = max(1, int(eval_every_steps))
        self.checkpoint_every_steps = max(1, int(checkpoint_every_steps))
        self.out_dir = Path(out_dir)
        self.best_mean_return = float("-inf")
        self.history = {"step": [], "mean_return": [], "success_rate": []}

    def _on_training_start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)
        if step > 0 and step % self.eval_every_steps == 0:
            metrics = self.eval_fn(self.model)
            mr = metrics["mean_return"]
            sr = metrics["success_rate"]
            print(f"PPO warm-start eval timestep={step} mean_return={mr:.4f} success_rate={sr:.4f}")
            self.history["step"].append(step)
            self.history["mean_return"].append(mr)
            self.history["success_rate"].append(sr)
            if mr > self.best_mean_return:
                self.best_mean_return = mr
                self.model.save(str(self.out_dir / "best_model"))
                print(f"Saved new best checkpoint (mean_return={self.best_mean_return:.4f}).")

        if step > 0 and step % self.checkpoint_every_steps == 0:
            self.model.save(str(self.out_dir / f"checkpoint_step_{step}"))

        return True


def make_augmented_throw_env(env_cfg, *, frame_stack: int, include_prev_action: bool, seed: int | None = None):
    env = make_throw_env(env_cfg, seed=seed)
    return AugmentedThrowObservationWrapper(
        env,
        frame_stack=frame_stack,
        include_prev_action=include_prev_action,
    )


def _make_augmented_factory(env_cfg, *, frame_stack: int, include_prev_action: bool):
    def _fn():
        return make_augmented_throw_env(
            env_cfg,
            frame_stack=frame_stack,
            include_prev_action=include_prev_action,
        )

    return _fn


def build_train_vec_env(
    env_cfg,
    *,
    seed: int,
    n_envs: int,
    vec_env_type: str,
    run_dir: Path,
    frame_stack: int,
    include_prev_action: bool,
):
    factory = _make_augmented_factory(
        env_cfg,
        frame_stack=frame_stack,
        include_prev_action=include_prev_action,
    )
    vec_env_cls = SubprocVecEnv if vec_env_type == "subproc" else DummyVecEnv
    vec_env_kwargs = {"start_method": "spawn"} if vec_env_cls is SubprocVecEnv else None
    return make_vec_env(
        factory,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_dir=str(run_dir / "monitor"),
    )


def evaluate_ppo_model(model: PPO, env_cfg, *, seed: int, n_episodes: int, frame_stack: int, include_prev_action: bool):
    env = make_augmented_throw_env(
        env_cfg,
        frame_stack=frame_stack,
        include_prev_action=include_prev_action,
        seed=seed,
    )
    returns = []
    successes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_return = 0.0
        info = {}
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
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


def load_bc_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = ["hidden_size", "obs_dim", "action_dim", "include_prev_action", "frame_stack"]
    missing = [name for name in required if name not in cfg]
    if missing:
        raise KeyError(f"BC config missing required fields: {missing}")
    return cfg


def copy_bc_actor_weights(model: PPO, bc_state_dict: dict, *, obs_dim: int, hidden_size: int, action_dim: int) -> None:
    policy_net = model.policy.mlp_extractor.policy_net
    action_net = model.policy.action_net
    if isinstance(action_net, nn.Sequential):
        action_linear = action_net[0]
    else:
        action_linear = action_net

    expected_shapes = {
        "net.0.weight": (hidden_size, obs_dim),
        "net.0.bias": (hidden_size,),
        "net.2.weight": (hidden_size, hidden_size),
        "net.2.bias": (hidden_size,),
        "net.4.weight": (action_dim, hidden_size),
        "net.4.bias": (action_dim,),
    }
    for key, expected in expected_shapes.items():
        actual = tuple(bc_state_dict[key].shape)
        if actual != expected:
            raise ValueError(f"BC weight {key} has shape {actual}; expected {expected}.")

    if not isinstance(policy_net[0], nn.Linear) or not isinstance(policy_net[2], nn.Linear):
        raise TypeError("PPO policy_net architecture does not match the BC two-layer MLP.")
    if not isinstance(action_linear, nn.Linear):
        raise TypeError("PPO action_net must be a Linear layer for BC weight transfer.")

    with torch.no_grad():
        policy_net[0].weight.copy_(bc_state_dict["net.0.weight"])
        policy_net[0].bias.copy_(bc_state_dict["net.0.bias"])
        policy_net[2].weight.copy_(bc_state_dict["net.2.weight"])
        policy_net[2].bias.copy_(bc_state_dict["net.2.bias"])
        action_linear.weight.copy_(bc_state_dict["net.4.weight"])
        action_linear.bias.copy_(bc_state_dict["net.4.bias"])


def supervised_actor_pretrain(
    model: PPO,
    *,
    dataset_path: Path,
    frame_stack: int,
    include_prev_action: bool,
    batch_size: int,
    learning_rate: float,
    train_steps: int,
    device: torch.device,
) -> list[float]:
    if train_steps <= 0:
        return []

    obs, actions = load_expert_transitions(
        dataset_path,
        include_prev_action=include_prev_action,
        frame_stack=frame_stack,
    )
    expected_obs_dim = int(model.observation_space.shape[0])
    if obs.shape[1] != expected_obs_dim:
        raise ValueError(f"Dataset obs_dim={obs.shape[1]} but PPO obs_dim={expected_obs_dim}.")

    dataset = TensorDataset(
        torch.as_tensor(obs, dtype=torch.float32),
        torch.as_tensor(actions, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    actor_params = (
        list(model.policy.mlp_extractor.policy_net.parameters())
        + list(model.policy.action_net.parameters())
        + [model.policy.log_std]
    )
    optimizer = optim.Adam(actor_params, lr=learning_rate)
    criterion = nn.MSELoss()
    losses = []
    step = 0
    model.policy.train()

    while step < train_steps:
        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            features = model.policy.extract_features(batch_obs)
            latent_pi, _ = model.policy.mlp_extractor(features)
            pred_actions = model.policy.action_net(latent_pi)
            loss = criterion(pred_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            step += 1
            if step % 100 == 0 or step == train_steps:
                print(f"[ppo-bc] supervised_pretrain step={step}/{train_steps} loss={losses[-1]:.6f}", flush=True)
            if step >= train_steps:
                break

    return losses


def run_single_seed(
    env_cfg,
    exp_cfg: ExperimentConfig,
    *,
    seed: int,
    run_dir: Path,
    device: str,
    n_envs: int,
    vec_env_type: str,
    bc_policy_path: Path,
    bc_config_path: Path,
    dataset_path: Path,
    pretrain_steps: int,
    pretrain_batch_size: int,
    pretrain_learning_rate: float,
    initial_log_std: float,
):
    reseed(seed)
    ensure_dir(run_dir)

    bc_cfg = load_bc_config(bc_config_path)
    frame_stack = int(bc_cfg["frame_stack"])
    include_prev_action = bool(bc_cfg["include_prev_action"])
    hidden_size = int(bc_cfg["hidden_size"])
    obs_dim = int(bc_cfg["obs_dim"])
    action_dim = int(bc_cfg["action_dim"])

    vec_env = build_train_vec_env(
        env_cfg,
        seed=seed,
        n_envs=n_envs,
        vec_env_type=vec_env_type,
        run_dir=run_dir,
        frame_stack=frame_stack,
        include_prev_action=include_prev_action,
    )
    vec_env = VecMonitor(vec_env)

    policy_kwargs = {
        "net_arch": {"pi": [hidden_size, hidden_size], "vf": [hidden_size, hidden_size]},
        "activation_fn": nn.ReLU,
        "log_std_init": initial_log_std,
    }
    model = PPO(
        BCTanhActorCriticPolicy,
        vec_env,
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=str(run_dir / "tb"),
        n_steps=512,
        learning_rate=3e-5,
        clip_range=0.05,
        n_epochs=1,
        target_kl=0.1,
        policy_kwargs=policy_kwargs,
    )

    if int(model.observation_space.shape[0]) != obs_dim:
        raise ValueError(f"PPO obs_dim={model.observation_space.shape[0]} but BC obs_dim={obs_dim}.")
    if int(model.action_space.shape[0]) != action_dim:
        raise ValueError(f"PPO action_dim={model.action_space.shape[0]} but BC action_dim={action_dim}.")

    bc_state = torch.load(str(bc_policy_path), map_location=model.device)
    copy_bc_actor_weights(
        model,
        bc_state,
        obs_dim=obs_dim,
        hidden_size=hidden_size,
        action_dim=action_dim,
    )
    print(f"[ppo-bc] copied BC actor weights from {bc_policy_path}", flush=True)

    pretrain_losses = supervised_actor_pretrain(
        model,
        dataset_path=dataset_path,
        frame_stack=frame_stack,
        include_prev_action=include_prev_action,
        batch_size=pretrain_batch_size,
        learning_rate=pretrain_learning_rate,
        train_steps=pretrain_steps,
        device=model.device,
    )

    init_metrics = evaluate_ppo_model(
        model,
        env_cfg,
        seed=seed + 9000,
        n_episodes=exp_cfg.eval_episodes,
        frame_stack=frame_stack,
        include_prev_action=include_prev_action,
    )
    write_json(run_dir / "bc_initialized_eval.json", init_metrics)
    model.save(str(run_dir / "bc_initialized_model"))
    print(
        "[ppo-bc] initialized eval "
        f"mean_return={init_metrics['mean_return']:.4f} success_rate={init_metrics['success_rate']:.4f}",
        flush=True,
    )

    callback = PeriodicEvalSaveCallback(
        eval_fn=lambda m: evaluate_ppo_model(
            m,
            env_cfg,
            seed=seed + 1000,
            n_episodes=exp_cfg.eval_episodes,
            frame_stack=frame_stack,
            include_prev_action=include_prev_action,
        ),
        eval_every_steps=exp_cfg.eval_every_steps,
        checkpoint_every_steps=exp_cfg.checkpoint_every_steps,
        out_dir=run_dir,
    )

    if exp_cfg.total_timesteps > 0:
        model.learn(
            total_timesteps=exp_cfg.total_timesteps,
            callback=callback,
            reset_num_timesteps=True,
        )
        model.save(str(run_dir / "final_model"))
    else:
        print("[ppo-bc] total_timesteps=0; saved initialized checkpoint only.", flush=True)

    vec_env.close()

    hist = dict(callback.history)
    hist["best_mean_eval_return"] = float(callback.best_mean_return)
    hist["bc_initialized_eval"] = init_metrics
    hist["pretrain_loss"] = pretrain_losses
    write_json(run_dir / "eval_history.json", hist)
    write_json(
        run_dir / "warmstart_config.json",
        {
            "seed": seed,
            "bc_policy_path": str(bc_policy_path),
            "bc_config_path": str(bc_config_path),
            "dataset_path": str(dataset_path),
            "frame_stack": frame_stack,
            "include_prev_action": include_prev_action,
            "hidden_size": hidden_size,
            "initial_log_std": initial_log_std,
            "pretrain_steps": pretrain_steps,
            "pretrain_batch_size": pretrain_batch_size,
            "pretrain_learning_rate": pretrain_learning_rate,
            "env": env_cfg.__dict__,
        },
    )

    if callback.history["step"]:
        save_curve_plot(
            callback.history["step"],
            callback.history["success_rate"],
            title="BC-warmstarted PPO success rate vs timesteps",
            xlabel="timesteps",
            ylabel="success_rate",
            out_path=run_dir / "success_curve.png",
        )
        save_curve_plot(
            callback.history["step"],
            callback.history["mean_return"],
            title="BC-warmstarted PPO mean return vs timesteps",
            xlabel="timesteps",
            ylabel="mean_return",
            out_path=run_dir / "return_curve.png",
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-every-steps", type=int, default=8192)
    parser.add_argument("--checkpoint-every-steps", type=int, default=65_536)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-root", type=str, default="artifacts")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--vec-env-type",
        type=str,
        default="dummy",
        choices=["dummy", "subproc"],
        help="Default dummy is safest on Windows; subproc helps on Linux-heavy machines.",
    )
    parser.add_argument("--bc-policy-path", type=str, default="artifacts/bc/best_bc_policy.pt")
    parser.add_argument("--bc-config-path", type=str, default="artifacts/bc/bc_config.json")
    parser.add_argument("--dataset-path", type=str, default="data/expert_demonstrations.npz")
    parser.add_argument("--pretrain-steps", type=int, default=5_000)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--initial-log-std",
        type=float,
        default=-4.0,
        help="Initial PPO Gaussian log std; lower values preserve the BC behavior early in RL.",
    )
    parser.add_argument(
        "--terminate-on-score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="End the episode when the ball scores.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="FetchPickAndPlace-v2",
        help="Gymnasium Robotics Fetch env id. Use Dense variant if you want distance shaping.",
    )
    parser.add_argument(
        "--floor-penalty",
        type=float,
        default=500.0,
        help=(
            "Positive penalty applied on the floor-termination step when the ball drops without "
            "scoring. Discourages the BC-warmstarted policy from dumping the ball to escape the "
            "-1/step sparse reward; keeps it doing what BC already showed works. Pass 0.0 to disable."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    resolved_device = args.device
    if args.device == "auto":
        resolved_device = "cuda" if str(select_device()) == "cuda" else "cpu"

    env_cfg = replace(
        make_dense_only_env_config(),
        goal_bonus=50.0,
        env_id=args.env_id,
        terminate_on_score=args.terminate_on_score,
        floor_penalty=args.floor_penalty,
    )
    exp_cfg = ExperimentConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        eval_every_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        checkpoint_every_steps=args.checkpoint_every_steps,
        n_seeds=1,
        output_root=args.output_root,
    )
    run_dir = ensure_dir(exp_cfg.output_root_path / "rl" / "bc_warmstart" / f"seed_{args.seed}")

    print(f"[ppo-bc] device={resolved_device}", flush=True)
    print(f"[ppo-bc] run_dir={run_dir}", flush=True)
    print(f"[ppo-bc] env_id={args.env_id} n_envs={args.n_envs} vec_env={args.vec_env_type}", flush=True)
    print(f"[ppo-bc] floor_penalty={args.floor_penalty}", flush=True)

    run_single_seed(
        env_cfg,
        exp_cfg,
        seed=args.seed,
        run_dir=run_dir,
        device=resolved_device,
        n_envs=args.n_envs,
        vec_env_type=args.vec_env_type,
        bc_policy_path=Path(args.bc_policy_path),
        bc_config_path=Path(args.bc_config_path),
        dataset_path=Path(args.dataset_path),
        pretrain_steps=args.pretrain_steps,
        pretrain_batch_size=args.pretrain_batch_size,
        pretrain_learning_rate=args.pretrain_learning_rate,
        initial_log_std=args.initial_log_std,
    )

    write_json(
        run_dir.parent / "summary.json",
        {"runs": [{"seed": args.seed, "eval_history_path": str(run_dir / "eval_history.json")}]},
    )


if __name__ == "__main__":
    main()
