from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen = True)
class EnvironmentConfig:
    env_id: str = "FetchPickAndPlace-v2"
    max_episode_steps: int = 200
    throw_overclock_factor: float = 3.0
    override_reward_on_score: bool = False
    score_reward: float = 0.0
    goal_bonus: float = 0.0
    terminate_on_score: bool = False


@dataclass(frozen = True)
class ExperimentConfig:
    seed: int = 7
    total_timesteps: int = 200_000
    eval_every_steps: int = 20_000
    eval_episodes: int = 20
    checkpoint_every_steps: int = 20_000
    n_seeds: int = 3
    run_name: str = "ppo_baseline"
    output_root: str = "artifacts"

    @property
    def output_root_path(self) -> Path:
        return Path(self.output_root)


@dataclass(frozen = True)
class BCConfig:
    batch_size: int = 256
    learning_rate: float = 3e-4
    train_steps: int = 30_000
    val_split: float = 0.2
    early_stop_patience: int = 30
    eval_every_steps: int = 200
    hidden_size: int = 256


def make_dense_only_env_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        override_reward_on_score = False,
        goal_bonus = 0.0,
        terminate_on_score = False,
    )


def make_dense_plus_bonus_env_config(goal_bonus: float = 5.0) -> EnvironmentConfig:
    return EnvironmentConfig(
        override_reward_on_score = False,
        goal_bonus = goal_bonus,
        terminate_on_score = False,
    )
