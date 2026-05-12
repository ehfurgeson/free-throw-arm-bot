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
    # End episode when ball center drops below this z (meters); None disables.
    terminate_ball_below_z: float | None = 0.1
    # If set, the wrapper overrides the env's randomized object spawn to this
    # fixed (x, y, z) on every reset. This removes the pickup-location
    # variation so BC only has to learn one canonical approach + throw motion.
    # Pickup itself was never the project's core challenge (the goal is to
    # shoot a basket), so pinning this is consistent with the project framing.
    # The z (0.45) is intentionally ~3 cm above the table surface so the ball
    # drops under gravity during the policy's pre-grasp idle phase rather
    # than starting interpenetrated with the table mesh.
    # Set to None to restore the original randomized spawn.
    fixed_object_position: tuple[float, float, float] | None = (1.3, 0.75, 0.45)


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
    # Number of consecutive env observations stacked as input to the BC policy.
    # K>1 lets the network recover the expert FSM's elapsed-time counters
    # (which gate hover->descend->grasp_wait transitions) from the recent
    # observation window.
    frame_stack: int = 4


def make_dense_only_env_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        override_reward_on_score = False,
        goal_bonus = 0.0,
        terminate_on_score = False,
    )


def make_dense_plus_bonus_env_config(goal_bonus: float = 50.0) -> EnvironmentConfig:
    return EnvironmentConfig(
        override_reward_on_score = False,
        goal_bonus = goal_bonus,
        terminate_on_score = False,
    )
