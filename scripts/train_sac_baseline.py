import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.her import HerReplayBuffer

# Allow running as `python scripts/train_sac_baseline.py` from repo root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.experiment_config import ExperimentConfig, make_dense_only_env_config
from scripts.training_utils import (
    ensure_dir,
    evaluate_policy_callable,
    make_throw_env,
    reseed,
    save_curve_plot,
    select_device,
    write_json,
)


def _make_throw_factory(env_cfg):
    """Callable with no arguments; SB3's make_vec_env applies per-env seeds."""

    def _fn():
        return make_throw_env(env_cfg)

    return _fn


def build_train_vec_env(
    env_cfg,
    *,
    seed: int,
    n_envs: int,
    vec_env_type: str,
    run_dir: Path,
):
    factory = _make_throw_factory(env_cfg)
    vec_env_cls = SubprocVecEnv if vec_env_type == "subproc" else DummyVecEnv
    vec_env_kwargs = {"start_method": "spawn"} if vec_env_cls is SubprocVecEnv else None
    monitor_dir = str(run_dir / "monitor")
    return make_vec_env(
        factory,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_dir=monitor_dir,
    )


def _parse_checkpoint_step(path: Path) -> int | None:
    stem = path.stem
    prefix = "checkpoint_step_"
    if stem.startswith(prefix) and stem[len(prefix) :].isdigit():
        return int(stem[len(prefix) :])
    return None


def find_resume_path(run_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    if not run_dir.is_dir():
        return None

    steps_paths: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint_step_*.zip"):
        n = _parse_checkpoint_step(path)
        if n is not None:
            steps_paths.append((n, path))
    if steps_paths:
        return max(steps_paths, key=lambda t: t[0])[1]

    for name in ("final_model.zip", "best_model.zip"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


class PeriodicEvalSaveCallback(BaseCallback):
    """Periodic deterministic eval + save best checkpoint (same pattern as PPO baseline)."""

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
            print(f"SAC periodic eval timestep={step} mean_return={mr:.4f} success_rate={sr:.4f}")
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


def run_single_seed(
    env_cfg,
    exp_cfg: ExperimentConfig,
    seed: int,
    run_dir: Path,
    device: str,
    n_envs: int,
    vec_env_type: str,
    resume_checkpoint: str | None,
    sac_learning_rate: float,
    sac_gamma: float,
    sac_batch_size: int,
    sac_tau: float,
    her_n_sampled_goal: int,
    her_goal_selection_strategy: str,
):
    reseed(seed)
    ensure_dir(run_dir)
    vec_env = build_train_vec_env(
        env_cfg,
        seed=seed,
        n_envs=n_envs,
        vec_env_type=vec_env_type,
        run_dir=run_dir,
    )
    vec_env = VecMonitor(vec_env)

    eval_fn = lambda model: evaluate_policy_callable(  # noqa: E731
        policy_fn=lambda obs: model.predict(obs, deterministic=True)[0],
        env_cfg=env_cfg,
        seed=seed + 1000,
        n_episodes=exp_cfg.eval_episodes,
    )
    callback = PeriodicEvalSaveCallback(
        eval_fn=eval_fn,
        eval_every_steps=exp_cfg.eval_every_steps,
        checkpoint_every_steps=exp_cfg.checkpoint_every_steps,
        out_dir=run_dir,
    )

    resume_path = find_resume_path(run_dir, resume_checkpoint)
    # HER must not sample until at least one full episode is in the buffer (SB3 runtime check).
    learning_starts = max(1_000, env_cfg.max_episode_steps + 1)
    common_kwargs = dict(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=str(run_dir / "tb"),
        learning_rate=sac_learning_rate,
        gamma=sac_gamma,
        batch_size=sac_batch_size,
        tau=sac_tau,
        learning_starts=learning_starts,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=her_n_sampled_goal,
            goal_selection_strategy=her_goal_selection_strategy,
        ),
        policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn=nn.ReLU),
    )

    if resume_path is not None:
        print(f"Resuming from: {resume_path}")
        model = SAC.load(str(resume_path), env=vec_env, device=device)
        # Checkpoints may store SB3 defaults; HER still needs this above max episode length.
        model.learning_starts = learning_starts
        reset_num_timesteps = False
    else:
        print("Starting new SAC + HER model (no resume checkpoint).")
        model = SAC(**common_kwargs)
        reset_num_timesteps = True

    model.learn(
        total_timesteps=exp_cfg.total_timesteps,
        callback=callback,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(str(run_dir / "final_model"))
    vec_env.close()

    hist = dict(callback.history)
    hist["best_mean_eval_return"] = float(callback.best_mean_return)
    write_json(run_dir / "eval_history.json", hist)
    if callback.history["step"]:
        save_curve_plot(
            callback.history["step"],
            callback.history["success_rate"],
            title="SAC success rate vs timesteps",
            xlabel="timesteps",
            ylabel="success_rate",
            out_path=run_dir / "success_curve.png",
        )
        save_curve_plot(
            callback.history["step"],
            callback.history["mean_return"],
            title="SAC mean return vs timesteps",
            xlabel="timesteps",
            ylabel="mean_return",
            out_path=run_dir / "return_curve.png",
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-every-steps", type=int, default=10_000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=65_536)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-root", type=str, default="artifacts")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument(
        "--vec-env-type",
        type=str,
        default="dummy",
        choices=["dummy", "subproc"],
        help="Default dummy is safest on Windows; subproc helps on Linux-heavy machines.",
    )
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument(
        "--terminate-on-score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="End the episode when the ball scores (matches PPO throw wrapper behavior).",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="FetchPickAndPlaceDense-v2",
        help="Use FetchPickAndPlaceDense-v2 for negative-distance shaping (recommended with HER).",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--her-n-sampled-goal", type=int, default=4)
    parser.add_argument(
        "--her-goal-selection-strategy",
        type=str,
        default="future",
        choices=["future", "episode", "final"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    resolved_device = args.device
    if args.device == "auto":
        resolved_device = "cuda" if str(select_device()) == "cuda" else "cpu"
    print(f"SAC device: {resolved_device}")
    print(f"Vec envs: {args.n_envs} ({args.vec_env_type})")
    print(f"Seed: {args.seed}")

    env_cfg = replace(
        make_dense_only_env_config(),
        env_id=args.env_id,
        terminate_on_score=args.terminate_on_score,
    )

    exp_cfg = ExperimentConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        eval_every_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        checkpoint_every_steps=args.checkpoint_every_steps,
        n_seeds=1,
        output_root=args.output_root,
        run_name="sac_baseline",
    )
    run_dir = ensure_dir(exp_cfg.output_root_path / "rl" / "sac_dense_only" / f"seed_{args.seed}")

    run_single_seed(
        env_cfg=env_cfg,
        exp_cfg=exp_cfg,
        seed=args.seed,
        run_dir=run_dir,
        device=resolved_device,
        n_envs=args.n_envs,
        vec_env_type=args.vec_env_type,
        resume_checkpoint=args.resume_checkpoint,
        sac_learning_rate=args.learning_rate,
        sac_gamma=args.gamma,
        sac_batch_size=args.batch_size,
        sac_tau=args.tau,
        her_n_sampled_goal=args.her_n_sampled_goal,
        her_goal_selection_strategy=args.her_goal_selection_strategy,
    )

    write_json(
        run_dir.parent / "summary.json",
        {"runs": [{"seed": args.seed, "eval_history_path": str(run_dir / "eval_history.json")}]},
    )


if __name__ == "__main__":
    main()
